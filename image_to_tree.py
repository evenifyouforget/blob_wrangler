import argparse
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops
import random
import time # Import the time module


class GeneratedRectangle:
    """
    A simple class to hold properties of a generated rectangle.
    Coordinates (x, y) refer to the top-left corner, and (width, height)
    are the dimensions. These coordinates typically represent the center
    in the Fantastic Contraption Markup Language (FCML).
    """
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        return f"Rectangle(x={self.x:.2f}, y={self.y:.2f}, width={self.width:.2f}, height={self.height:.2f})"


class GeneratedLine:
    """
    Represents a line with thickness, either horizontal or vertical.
    Coordinates (x1, y1) and (x2, y2) are the global image coordinates
    of its endpoints.
    """
    def __init__(self, x1, y1, x2, y2, thickness):
        # Ensure x1 <= x2 and y1 <= y2 for consistent internal representation
        # This simplifies length calculations and checks.
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.thickness = thickness

        # Determine orientation based on coordinate differences.
        # Lines must be strictly horizontal or vertical.
        # A small tolerance (0.1) is used for floating point comparisons.
        if abs(self.x1 - self.x2) < 0.1 and abs(self.y1 - self.y2) < 0.1:
            # It's a point, effectively zero length, which is invalid.
            raise ValueError(f"Line cannot have zero length: ({x1},{y1})-({x2},{y2})")
        elif abs(self.y1 - self.y2) < 0.1:
            # Y-coordinates are the same (within tolerance), so it's horizontal.
            self.orientation = 'horizontal'
        elif abs(self.x1 - self.x2) < 0.1:
            # X-coordinates are the same (within tolerance), so it's vertical.
            self.orientation = 'vertical'
        else:
            # Diagonal lines are not supported.
            raise ValueError(f"Line must be horizontal or vertical, got ({x1},{y1})-({x2},{y2})")

    def __str__(self):
        return (f"Line({self.x1:.2f}, {self.y1:.2f}) to ({self.x2:.2f}, {self.y2:.2f}), "
                f"Thickness={self.thickness:.2f}, Orientation={self.orientation})")

    def to_rectangle(self):
        """
        Converts the line to a GeneratedRectangle object representing its
        unpadded bounding box. The (x,y) of the rectangle is its top-left corner.
        """
        if self.orientation == 'horizontal':
            return GeneratedRectangle(
                self.x1,
                self.y1 - self.thickness / 2, # Top-left Y for rectangle
                self.x2 - self.x1,            # Width
                self.thickness                # Height
            )
        else: # vertical
            return GeneratedRectangle(
                self.x1 - self.thickness / 2, # Top-left X for rectangle
                self.y1,                      # Top-left Y for rectangle
                self.thickness,               # Width
                self.y2 - self.y1             # Height
            )

    def intersects_rectangle(self, rect):
        """Checks if this line (as an unpadded rectangle) intersects another rectangle."""
        my_rect = self.to_rectangle()
        return (my_rect.x < rect.x + rect.width and
                my_rect.x + my_rect.width > rect.x and
                my_rect.y < rect.y + rect.height and
                my_rect.y + my_rect.height > rect.y)


def _get_random_green_point(local_blob_mask, min_row, min_col):
    """
    Picks a random green pixel within the local blob mask and converts its
    coordinates to global image coordinates.
    """
    green_pixels = np.argwhere(local_blob_mask)
    if not green_pixels.size > 0:
        return None
    
    # Pick a random green pixel (row, col) in local mask coordinates
    random_idx = random.randint(0, len(green_pixels) - 1)
    local_y, local_x = green_pixels[random_idx]

    # Convert to global coordinates, considering the center of the pixel for line placement.
    global_x = min_col + local_x + 0.5
    global_y = min_row + local_y + 0.5
    return global_x, global_y


def _distance_between_rects(rect1, rect2):
    """
    Calculates the minimum distance between the edges of two axis-aligned rectangles.
    Returns 0 if they overlap or touch.
    """
    # Calculate horizontal gap between rectangles
    dx = max(0.0, rect1.x - (rect2.x + rect2.width), rect2.x - (rect1.x + rect1.width))
    # Calculate vertical gap between rectangles
    dy = max(0.0, rect1.y - (rect2.y + rect2.height), rect2.y - (rect1.y + rect1.height))

    # The minimum distance is the maximum of the horizontal and vertical gaps.
    # If they overlap in one dimension, that dimension's gap is 0, correctly
    # reflecting clearance.
    return max(dx, dy)


def _shares_endpoint(line, point_x, point_y, tolerance=0.5):
    """
    Checks if a given point (point_x, point_y) is one of the line's endpoints.
    Uses a small tolerance for floating point comparisons to handle precision issues.
    """
    is_endpoint_1 = (abs(line.x1 - point_x) < tolerance and abs(line.y1 - point_y) < tolerance)
    is_endpoint_2 = (abs(line.x2 - point_x) < tolerance and abs(line.y2 - point_y) < tolerance)
    return is_endpoint_1 or is_endpoint_2


def _check_line_segment_validity(start_x, start_y, end_x, end_y, thickness, local_blob_mask, min_row, min_col):
    """
    Checks if a proposed line segment (with its given thickness) remains entirely
    within the green blob mask. This performs a simplified pixel sampling check.

    Returns:
        tuple[bool, str]: A boolean indicating validity and a reason string if invalid.
    """
    # Convert global coordinates to local blob mask coordinates
    local_x1 = int(start_x - min_col)
    local_y1 = int(start_y - min_row)
    local_x2 = int(end_x - min_col)
    local_y2 = int(end_y - min_row)

    mask_height, mask_width = local_blob_mask.shape

    # Determine line orientation based on whether Y-coordinates are approximately the same.
    is_horizontal = abs(start_y - end_y) < 0.1
    
    # Calculate bounds for the line's thickness in local coordinates.
    if is_horizontal:
        center_y_local = int(start_y - min_row)
        min_local_y_thick = int(center_y_local - thickness / 2)
        max_local_y_thick = int(center_y_local + thickness / 2) # Exclusive upper bound
        
        # Clamp thickness bounds to mask dimensions to prevent out-of-bounds access.
        min_local_y_thick = max(0, min_local_y_thick)
        max_local_y_thick = min(mask_height, max_local_y_thick)

        # Iterate along the line's length (X-axis) and check all pixels within its thickness.
        for lx in range(min(local_x1, local_x2), max(local_x1, local_x2) + 1):
            for ly in range(min_local_y_thick, max_local_y_thick):
                # Ensure current pixel (lx, ly) is within mask bounds AND is part of the green blob.
                if not (0 <= lx < mask_width and 0 <= ly < mask_height and local_blob_mask[ly, lx]):
                    return False, "Outside blob (horizontal line segment)"
        return True, "Within blob (horizontal line segment)"
    else: # Vertical
        center_x_local = int(start_x - min_col)
        min_local_x_thick = int(center_x_local - thickness / 2)
        max_local_x_thick = int(center_x_local + thickness / 2) # Exclusive upper bound
        
        # Clamp thickness bounds to mask dimensions.
        min_local_x_thick = max(0, min_local_x_thick)
        max_local_x_thick = min(mask_width, max_local_x_thick)

        # Iterate along the line's length (Y-axis) and check all pixels within its thickness.
        for ly in range(min(local_y1, local_y2), max(local_y1, local_y2) + 1):
            for lx in range(min_local_x_thick, max_local_x_thick):
                # Ensure current pixel (lx, ly) is within mask bounds AND is part of the green blob.
                if not (0 <= lx < mask_width and 0 <= ly < mask_height and local_blob_mask[ly, lx]):
                    return False, "Outside blob (vertical line segment)"
        return True, "Within blob (vertical line segment)"


def _can_add_line_segment_within_blob(new_line, local_blob_mask, min_row, min_col):
    """
    Convenience wrapper to check if a new line segment stays within the green blob mask.
    """
    return _check_line_segment_validity(new_line.x1, new_line.y1, new_line.x2, new_line.y2, new_line.thickness, local_blob_mask, min_row, min_col)


def _can_add_line_segment(new_line, existing_lines, start_point_x, start_point_y, min_spacing_val, local_blob_mask, min_row, min_col, line_padding):
    """
    Checks if a new line segment can be validly added to the existing tree structure.

    Args:
        new_line (GeneratedLine): The candidate line segment to add.
        existing_lines (list[GeneratedLine]): List of already generated lines.
        start_point_x (float): X-coordinate of the point from which 'new_line' is trying to extend/branch.
        start_point_y (float): Y-coordinate of the point from which 'new_line' is trying to extend/branch.
        min_spacing_val (float): Minimum required spacing between non-connected padded lines.
        local_blob_mask: The mask of the green blob.
        min_row, min_col: Bounding box for the green blob.
        line_padding (float): The padding value to be applied for collision checks.

    Returns:
        tuple[bool, str]: A boolean indicating validity and a reason string if invalid.
    """
    # 1. Check if the proposed line segment (unpadded) stays within the green blob mask.
    is_within_blob, blob_reason = _can_add_line_segment_within_blob(new_line, local_blob_mask, min_row, min_col)
    if not is_within_blob:
        return False, blob_reason

    # 2. If this is the very first line being placed, it is always valid (assuming it's in blob).
    if not existing_lines:
        return True, "Valid (first segment)"
    
    # Create a padded version of the new_line's rectangle for collision checks.
    new_rect_padded = new_line.to_rectangle()
    if new_line.orientation == 'horizontal':
        new_rect_padded.x -= line_padding
        new_rect_padded.width += 2 * line_padding
    else: # vertical
        new_rect_padded.y -= line_padding
        new_rect_padded.height += 2 * line_padding

    # Identify existing lines that share the 'start_point' of the new line candidate.
    # These are considered "neighbors" and have different rules for collision.
    neighbors = []
    for existing_line in existing_lines:
        if _shares_endpoint(existing_line, start_point_x, start_point_y, tolerance=0.5):
            neighbors.append(existing_line)
    
    # 3. Apply collision, spacing, and redundancy rules against all existing lines.
    for existing_line in existing_lines:
        # Create a padded version of the existing line's rectangle for collision checks.
        existing_rect_padded = existing_line.to_rectangle()
        if existing_line.orientation == 'horizontal':
            existing_rect_padded.x -= line_padding
            existing_rect_padded.width += 2 * line_padding
        else: # vertical
            existing_rect_padded.y -= line_padding
            existing_rect_padded.height += 2 * line_padding

        # --- Redundancy Check for all collinear lines (including neighbors) ---
        # This prevents duplicate lines, lines contained within others, or lines containing others.
        # This is critical to avoid the "stacking" issue.
        if (new_line.orientation == existing_line.orientation and
            abs(new_line.thickness - existing_line.thickness) < 0.1): # Only compare if collinear and same thickness
            
            # Check if new_line is fully contained within existing_line (or is identical).
            if new_line.orientation == 'horizontal':
                if (new_line.x1 >= existing_line.x1 - 0.1 and new_line.x2 <= existing_line.x2 + 0.1 and
                    abs(new_line.y1 - existing_line.y1) < 0.1):
                    return False, "Redundant (new line contained in existing collinear line)"
                # Check if existing_line is fully contained within new_line.
                if (existing_line.x1 >= new_line.x1 - 0.1 and existing_line.x2 <= new_line.x2 + 0.1 and
                    abs(new_line.y1 - existing_line.y1) < 0.1):
                    return False, "Redundant (new line contains existing collinear line)"
            else: # vertical
                if (new_line.y1 >= existing_line.y1 - 0.1 and new_line.y2 <= existing_line.y2 + 0.1 and
                    abs(new_line.x1 - existing_line.x1) < 0.1):
                    return False, "Redundant (new line contained in existing collinear line)"
                if (existing_line.y1 >= new_line.y1 - 0.1 and existing_line.y2 <= new_line.y2 + 0.1 and
                    abs(new_line.x1 - existing_line.x1) < 0.1):
                    return False, "Redundant (new line contains existing collinear line)"
            
            # Additional check for near-exact duplicates, especially when floating point makes perfect containment tricky.
            if (abs(new_line.x1 - existing_line.x1) < 0.1 and abs(new_line.y1 - existing_line.y1) < 0.1 and
                abs(new_line.x2 - existing_line.x2) < 0.1 and abs(new_line.y2 - existing_line.y2) < 0.1):
                return False, "Redundant (near-exact duplicate line)"

        # If the existing_line is a direct neighbor, it means the new line is attempting to extend from it.
        # We allow overlap/touching for neighbors because they are conceptually connected.
        # The redundancy checks above should catch unintended collinear overlaps with neighbors.
        if existing_line in neighbors:
            continue

        # --- Collision and Spacing Checks for non-neighboring lines (using padded dimensions) ---
        # Rule: If new padded rectangle overlaps an existing padded rectangle that is NOT a neighbor.
        if (new_rect_padded.x < existing_rect_padded.x + existing_rect_padded.width and
            new_rect_padded.x + new_rect_padded.width > existing_rect_padded.x and
            new_rect_padded.y < existing_rect_padded.y + existing_rect_padded.height and
            new_rect_padded.y + new_rect_padded.height > existing_rect_padded.y):
            return False, "Overlap with non-neighboring padded line"

        # Rule: If new padded rectangle is too close to an existing padded rectangle that is NOT a neighbor.
        dist = _distance_between_rects(new_rect_padded, existing_rect_padded)
        if dist < min_spacing_val:
            return False, f"Violates min_spacing ({dist:.2f} < {min_spacing_val:.2f}) with non-neighboring padded line"

    # If all checks pass, the new line segment is valid.
    return True, "Valid (connected and spaced with non-redundant lines)"


def _perform_line_cut(generated_lines, min_segment_length_after_cut, local_blob_mask, min_row, min_col, min_spacing, line_padding):
    """
    Attempts to cut an existing line segment into two smaller, valid segments.

    Args:
        generated_lines (list): The current list of GeneratedLine objects.
        min_segment_length_after_cut (float): The minimum allowed length for each new segment
                                               created by the cut (e.g., 80 units if thickness is 40 and multiplier is 2).
        local_blob_mask: The mask of the current blob.
        min_row, min_col: Bounding box for local_blob_mask.
        min_spacing: Minimum spacing value for validity checks of new segments.
        line_padding (float): Amount of padding, needed for `_can_add_line_segment`.

    Returns:
        bool: True if a cut was successfully performed, False otherwise.
    """
    if not generated_lines:
        return False

    line_to_cut = random.choice(generated_lines)
    
    # Calculate the original length of the line to be cut.
    if line_to_cut.orientation == 'horizontal':
        original_length = abs(line_to_cut.x2 - line_to_cut.x1)
    else: # vertical
        original_length = abs(line_to_cut.y2 - line_to_cut.y1)

    # A line can only be cut if its length is at least twice the minimum segment length.
    # This ensures that two valid segments can be created.
    if original_length < 2 * min_segment_length_after_cut:
        return False # Line is too short to be cut into two valid segments

    # Determine the valid range for the cut point along the line.
    # The cut point must ensure both new segments meet the minimum length.
    min_cut_pos_from_start = min_segment_length_after_cut
    max_cut_pos_from_start = original_length - min_segment_length_after_cut

    # If the valid range is inverted or zero, no valid cut point exists.
    if max_cut_pos_from_start <= min_cut_pos_from_start:
        return False
    
    # Choose a random cut point within the valid range.
    cut_pos_along_line = random.uniform(min_cut_pos_from_start, max_cut_pos_from_start)

    new_line1 = None
    new_line2 = None

    # Calculate global coordinates for the cut point and create the two new line segments.
    if line_to_cut.orientation == 'horizontal':
        cut_x = min(line_to_cut.x1, line_to_cut.x2) + cut_pos_along_line
        cut_y = line_to_cut.y1 # Y-coordinate remains the same
        new_line1 = GeneratedLine(line_to_cut.x1, line_to_cut.y1, cut_x, cut_y, line_to_cut.thickness)
        new_line2 = GeneratedLine(cut_x, cut_y, line_to_cut.x2, line_to_cut.y2, line_to_cut.thickness)
    else: # vertical
        cut_y = min(line_to_cut.y1, line_to_cut.y2) + cut_pos_along_line
        cut_x = line_to_cut.x1 # X-coordinate remains the same
        new_line1 = GeneratedLine(line_to_cut.x1, line_to_cut.y1, cut_x, cut_y, line_to_cut.thickness)
        new_line2 = GeneratedLine(cut_x, cut_y, line_to_cut.x2, line_to_cut.y2, line_to_cut.thickness)
    
    # Temporarily remove the original line to validate the two new segments against the rest of the tree.
    temp_existing_lines = [l for l in generated_lines if l != line_to_cut]

    # Validate both new segments. The 'start_point' for validation is their respective connected end.
    is_valid1, _ = _can_add_line_segment(new_line1, temp_existing_lines, new_line1.x1, new_line1.y1, min_spacing, local_blob_mask, min_row, min_col, line_padding)
    is_valid2, _ = _can_add_line_segment(new_line2, temp_existing_lines, new_line2.x2, new_line2.y2, min_spacing, local_blob_mask, min_row, min_col, line_padding)

    # If both new segments are valid, replace the original line with them.
    if is_valid1 and is_valid2:
        generated_lines.remove(line_to_cut)
        generated_lines.append(new_line1)
        generated_lines.append(new_line2)
        return True
    
    return False


def process_green_blob(blob_id, region, fixed_dimension_size, min_spacing, max_iterations, min_line_length_multiplier, max_line_length_multiplier, initial_segment_attempts, line_padding, horizontal_bias, cut_probability, min_segment_length_after_cut_multiplier):
    """
    Processes a single contiguous green region (blob) to generate a tree-like structure
    of thick lines using a random walk algorithm with growth and optional cut operations.

    Args:
        blob_id (int): A unique identifier for this green blob.
        region (skimage.measure._regionprops.RegionProperties): The properties object for the blob.
        fixed_dimension_size (float): The desired thickness for the generated lines.
        min_spacing (float): The minimum spacing to enforce between non-connected orthogonal rectangles.
        max_iterations (int): Maximum attempts to add or cut line segments within this blob.
        min_line_length_multiplier (float): Multiplier for `fixed_dimension_size` to determine
                                            the minimum length for newly grown line segments.
        max_line_length_multiplier (float): Multiplier for `fixed_dimension_size` to determine
                                            the maximum length for newly grown line segments.
        initial_segment_attempts (int): Number of attempts to place the very first line segment for a blob.
        line_padding (float): Amount to extend each line segment on both ends (affects final rectangle size).
        horizontal_bias (float): Bias towards horizontal lines (0.0-1.0, 0.5 is neutral).
        cut_probability (float): Probability (0.0-1.0) of attempting a line cut operation
                                 instead of a line growth operation in an iteration.
        min_segment_length_after_cut_multiplier (float): Multiplier for `fixed_dimension_size` to determine
                                                         the minimum length of *each* segment resulting from a cut.

    Returns:
        list[GeneratedRectangle]: A list of final rectangles generated for this blob, ready for FCML output.
    """
    print(f"\n--- Processing Green Blob #{blob_id} ---")
    min_row, min_col, max_row, max_col = region.bbox
    print(f"  Bounding Box (min_row, min_col, max_row, max_col): {region.bbox}")
    print(f"  Area (pixels): {region.area}")
    print(f"  Centroid (row, col): ({region.centroid[0]:.2f}, {region.centroid[1]:.2f})")
    print(f"  Line Thickness: {fixed_dimension_size} pixels")
    print(f"  Minimum Spacing Target: {min_spacing} pixels")
    print(f"  Horizontal Bias: {horizontal_bias}")
    print(f"  Cut Probability: {cut_probability}")
    print(f"  Min Segment Length After Cut Multiplier: {min_segment_length_after_cut_multiplier}")


    generated_lines = [] # Stores GeneratedLine objects as the tree grows
    local_blob_mask = region.image # Mask of the blob relative to its bounding box (True for green pixels)

    # Find a random starting point within the green blob for the initial segment.
    start_point = _get_random_green_point(local_blob_mask, min_row, min_col)
    if start_point is None:
        print("  Could not find a valid starting point within the green blob. Skipping.")
        return []

    initial_x, initial_y = start_point
    line_thickness = fixed_dimension_size
    # Calculate actual min/max lengths for new line growth and minimum length for cut segments.
    min_line_length = line_thickness * min_line_length_multiplier
    max_line_length = line_thickness * max_line_length_multiplier
    min_segment_length_after_cut = line_thickness * min_segment_length_after_cut_multiplier


    print(f"  Starting growth from ({initial_x:.2f}, {initial_y:.2f}) with max iterations {max_iterations}...")

    # Attempt to create the very first line segment for this blob.
    initial_segment_added = False
    for _ in range(initial_segment_attempts):
        try:
            # Apply horizontal bias when choosing the initial orientation.
            current_orientation = 'horizontal' if random.random() < horizontal_bias else 'vertical'
            
            direction_multiplier = random.choice([1, -1]) # Randomly choose positive or negative direction.
            length = random.uniform(min_line_length, max_line_length)

            if current_orientation == 'horizontal':
                end_x = initial_x + length * direction_multiplier
                initial_line = GeneratedLine(initial_x, initial_y, end_x, initial_y, line_thickness)
            else: # Vertical
                end_y = initial_y + length * direction_multiplier
                initial_line = GeneratedLine(initial_x, initial_y, initial_x, end_y, line_thickness)
            
            # Validate the initial line segment. No existing lines yet to check against.
            is_valid, _ = _can_add_line_segment(initial_line, [], initial_x, initial_y, min_spacing, local_blob_mask, min_row, min_col, line_padding)
            if is_valid:
                generated_lines.append(initial_line)
                initial_segment_added = True
                print(f"  Initial segment added: {initial_line}")
                break
        except ValueError:
            # Catch cases where line constructor might raise error (e.g., zero length after tiny movement).
            continue
    
    if not initial_segment_added:
        print("  Could not generate a valid initial segment. Skipping blob.")
        return []

    # Main growth and cut loop for generating the tree structure.
    start_time_blob = time.time()
    last_print_time = start_time_blob
    for i in range(max_iterations):
        # Print progress update every second.
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            progress_percent = (i / max_iterations) * 100
            print(f"  Blob #{blob_id} Progress: Iteration {i}/{max_iterations} ({progress_percent:.2f}%)")
            last_print_time = current_time

        # Randomly decide whether to grow a new line or cut an existing one, based on cut_probability.
        if generated_lines and random.random() < cut_probability:
            # Attempt a cut operation on an existing line.
            _perform_line_cut(generated_lines, min_segment_length_after_cut, local_blob_mask, min_row, min_col, min_spacing, line_padding)
        else:
            # Attempt a grow operation: extend from an existing line's endpoint.
            if not generated_lines: # Ensure there's something to extend from if cut_probability was 0 and initial failed
                continue

            # Pick a random point (endpoint) from an existing line in the tree.
            parent_line_from_tree = random.choice(generated_lines)
            start_point_candidates = [(parent_line_from_tree.x1, parent_line_from_tree.y1),
                                      (parent_line_from_tree.x2, parent_line_from_tree.y2)]
            current_x, current_y = random.choice(start_point_candidates)

            # Apply bias when picking the orientation for the new segment.
            chosen_orientation = 'horizontal' if random.random() < horizontal_bias else 'vertical'

            # Determine direction multiplier based on the chosen orientation.
            if chosen_orientation == 'horizontal':
                dir_x_mult = random.choice([1, -1])
                dir_y_mult = 0
            else: # vertical
                dir_x_mult = 0
                dir_y_mult = random.choice([1, -1])

            # Pick a random length for the new segment.
            length = random.uniform(min_line_length, max_line_length)

            # Calculate the proposed end point for the new segment.
            potential_end_x = current_x + length * dir_x_mult
            potential_end_y = current_y + length * dir_y_mult

            # Create the new candidate line segment.
            try:
                new_line_candidate = GeneratedLine(current_x, current_y, potential_end_x, potential_end_y, line_thickness)
            except ValueError:
                # Skip if the attempted line is invalid (e.g., zero length).
                continue

            # Validate the new segment against existing lines, enforcing spacing and redundancy rules.
            is_valid, reason = _can_add_line_segment(new_line_candidate, generated_lines, current_x, current_y, min_spacing, local_blob_mask, min_row, min_col, line_padding)
            
            if is_valid:
                generated_lines.append(new_line_candidate)
            # else:
                # Uncomment for debugging: print(f"  Rejected (iteration {i + 1}): {reason} for {new_line_candidate}")

    time_taken_blob = time.time() - start_time_blob
    print(f"  Blob #{blob_id} processing finished. Time taken: {time_taken_blob:.2f} seconds.")


    print(f"  Growth completed. Generated {len(generated_lines)} raw line segments.")

    # Merge collinear and connected lines to simplify the structure.
    print(f"  Attempting to merge {len(generated_lines)} generated lines...")
    merged_lines = merge_collinear_lines(generated_lines)
    print(f"  After merging, {len(merged_lines)} lines remain.")

    # Convert the final merged lines into GeneratedRectangle objects, applying padding.
    final_rectangles = []
    for line in merged_lines:
        rect = line.to_rectangle() # Get unpadded rectangle first

        # Apply padding to the rectangle dimensions and position.
        if line.orientation == 'horizontal':
            rect.x -= line_padding
            rect.width += 2 * line_padding
        else: # vertical
            rect.y -= line_padding
            rect.height += 2 * line_padding
            
        # Calculate center coordinates for FCML output.
        center_x = rect.x + rect.width / 2
        center_y = rect.y + rect.height / 2
        final_rectangles.append(GeneratedRectangle(center_x, center_y, rect.width, rect.height))


    print(f"  Final generated {len(final_rectangles)} rectangles for this blob.")
    return final_rectangles


def merge_collinear_lines(lines):
    """
    Merges collinear and overlapping/touching line segments into longer, single segments.
    This helps simplify the generated structure and reduce redundant output.

    Args:
        lines (list[GeneratedLine]): A list of GeneratedLine objects to merge.

    Returns:
        list[GeneratedLine]: A new list of merged GeneratedLine objects.
    """
    if not lines:
        return []

    # Separate lines by orientation for independent merging.
    horizontal_lines = [line for line in lines if line.orientation == 'horizontal']
    vertical_lines = [line for line in lines if line.orientation == 'vertical']

    merged_h_lines = []
    # Sort horizontal lines primarily by Y-coordinate, then by their starting X-coordinate.
    # This ordering facilitates efficient merging of adjacent or overlapping segments.
    horizontal_lines.sort(key=lambda l: (l.y1, l.x1))
    
    # Process and merge horizontal lines.
    if horizontal_lines:
        current_merged = horizontal_lines[0] # Initialize with the first line.
        for i in range(1, len(horizontal_lines)):
            line = horizontal_lines[i]
            
            # Check if the current line is collinear and has the same thickness as the `current_merged` segment.
            is_collinear_and_same_thickness = (
                abs(current_merged.y1 - line.y1) < 0.1 and      # Same Y-coordinate
                abs(current_merged.thickness - line.thickness) < 0.1 # Same thickness
            )
            
            # Check if the current line overlaps or touches the `current_merged` segment.
            # This allows merging even if there's a slight gap or overlap.
            overlaps_or_touches = (line.x1 <= current_merged.x2 + 0.1 and line.x2 >= current_merged.x1 - 0.1)

            if is_collinear_and_same_thickness and overlaps_or_touches:
                # If conditions met, extend `current_merged` to encompass the new line's extent.
                current_merged.x1 = min(current_merged.x1, line.x1)
                current_merged.x2 = max(current_merged.x2, line.x2)
            else:
                # If no merge, add the `current_merged` segment to the result list and start a new `current_merged`.
                merged_h_lines.append(current_merged)
                current_merged = line
        merged_h_lines.append(current_merged) # Add the last processed merged line.

    merged_v_lines = []
    # Sort vertical lines primarily by X-coordinate, then by their starting Y-coordinate.
    vertical_lines.sort(key=lambda l: (l.x1, l.y1))

    # Process and merge vertical lines (logic mirrors horizontal merging).
    if vertical_lines:
        current_merged = vertical_lines[0]
        for i in range(1, len(vertical_lines)):
            line = vertical_lines[i]
            
            is_collinear_and_same_thickness = (
                abs(current_merged.x1 - line.x1) < 0.1 and
                abs(current_merged.thickness - line.thickness) < 0.1
            )
            
            overlaps_or_touches = (line.y1 <= current_merged.y2 + 0.1 and line.y2 >= current_merged.y1 - 0.1)
            
            if is_collinear_and_same_thickness and overlaps_or_touches:
                current_merged.y1 = min(current_merged.y1, line.y1)
                current_merged.y2 = max(current_merged.y2, line.y2)
            else:
                merged_v_lines.append(current_merged)
                current_merged = line
        merged_v_lines.append(current_merged)

    # Return the combined list of merged horizontal and vertical lines.
    return merged_h_lines + merged_v_lines


def process_image_for_contraption(image_path, fixed_dimension_size, min_spacing, output_file, max_iterations, min_line_length_multiplier, max_line_length_multiplier, initial_segment_attempts, min_blob_area, line_padding, horizontal_bias, cut_probability, min_segment_length_after_cut_multiplier):
    """
    Main function to process an image and generate Fantastic Contraption
    level decorations based on green regions. It outputs the generated
    rectangles in FCML format.

    Args:
        image_path (str): The file path to the input image (e.g., PNG, JPEG).
        fixed_dimension_size (float): The fixed thickness for all generated lines/rectangles.
        min_spacing (float): The minimum spacing to enforce between non-connected padded rectangles.
        output_file (str): The path to an output file to save the generated rectangles in FCML format.
        max_iterations (int): Maximum number of attempts to add or cut line segments per green blob.
        min_line_length_multiplier (float): Multiplier for `fixed_dimension_size` to determine
                                            the minimum length for new line growth segments.
        max_line_length_multiplier (float): Multiplier for `fixed_dimension_size` to determine
                                            the maximum length for new line growth segments.
        initial_segment_attempts (int): Number of attempts to place the very first line segment for a blob.
        min_blob_area (int): Minimum pixel area for a green blob to be processed. Smaller blobs are skipped.
        line_padding (float): Amount to extend each line segment on both ends (e.g., 40.0 means 40 pixels added to each end).
        horizontal_bias (float): Bias towards horizontal line generation (0.0-1.0, 0.5 is neutral, >0.5 prefers horizontal).
        cut_probability (float): Probability (0.0-1.0) of attempting a line cutting operation
                                 instead of a line growth operation in an iteration.
        min_segment_length_after_cut_multiplier (float): Multiplier for `fixed_dimension_size` to determine
                                                         the minimum length of *each* segment resulting from a cut.
    """
    print(f"Received image for processing: {image_path}")
    all_generated_rectangles = [] # Collects all GeneratedRectangle objects from all processed blobs.

    overall_start_time = time.time() # Record start time for overall statistics.

    try:
        with Image.open(image_path) as img:
            print(f"Image successfully loaded. Format: {img.format}, Size: {img.size}")

            # Convert image to RGB mode if it's not already.
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("Image converted to RGB mode.")

            img_np = np.array(img)

            # Define the green color range for masking.
            lower_green = np.array([0, 100, 0])
            upper_green = np.array([100, 255, 100])

            # Create a boolean mask where green pixels are True.
            green_mask = np.all((img_np >= lower_green) & (img_np <= upper_green), axis=-1)
            # Label connected components in the green mask to identify individual blobs.
            labeled_mask = label(green_mask)
            # Extract properties for each labeled region (blob).
            regions = regionprops(labeled_mask)

            print(f"\nFound {len(regions)} contiguous green regions.")

            # Process each green blob independently.
            for i, region in enumerate(regions):
                # Only process blobs larger than the specified minimum area.
                if region.area > min_blob_area:
                    rects = process_green_blob(
                        blob_id=region.label,
                        region=region,
                        fixed_dimension_size=fixed_dimension_size,
                        min_spacing=min_spacing,
                        max_iterations=max_iterations,
                        min_line_length_multiplier=min_line_length_multiplier,
                        max_line_length_multiplier=max_line_length_multiplier,
                        initial_segment_attempts=initial_segment_attempts,
                        line_padding=line_padding,
                        horizontal_bias=horizontal_bias,
                        cut_probability=cut_probability,
                        min_segment_length_after_cut_multiplier=min_segment_length_after_cut_multiplier
                    )
                    all_generated_rectangles.extend(rects)
                else:
                    print(f"\n--- Skipping small blob #{region.label} with area {region.area} (less than {min_blob_area}) ---")

            if not regions:
                print("No green regions found within the specified color range in the image.")

        # Write all generated rectangles to the specified output file in FCML format.
        print(f"\nWriting {len(all_generated_rectangles)} rectangles to {output_file}...")
        with open(output_file, 'w') as f:
            for rect in all_generated_rectangles:
                # FCML format: SR x y width height (where x,y are CENTER coordinates).
                f.write(f"SR {rect.x:.2f} {rect.y:.2f} {rect.width:.2f} {rect.height:.2f}\n")
        print("Finished writing FCML file.")

    except FileNotFoundError:
        print(f"Error: The image file was not found at {image_path}. Please check the path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred while processing the image: {e}")
    
    overall_time_taken = time.time() - overall_start_time # Calculate total execution time.
    print(f"\nOverall processing completed. Total time taken: {overall_time_taken:.2f} seconds.")


if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Generate Fantastic Contraption level decorations from an image by extracting "
                    "green regions and growing tree-like structures within them. Outputs in FCML format."
    )
    parser.add_argument(
        "-i", "--image-file",
        type=str,
        required=True,
        help="The path to the input image file (e.g., PNG, JPEG) containing green regions."
    )
    parser.add_argument(
        "-t", "--fixed-dimension-size",
        type=float,
        default=40.0,
        help="The fixed thickness (in pixels) for all generated lines/rectangles. Default is 40."
    )
    parser.add_argument(
        "-s", "--min-spacing",
        type=float,
        default=40.0,
        help="The minimum clear space (in pixels) to enforce between non-connected padded rectangles. Default is 40."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        required=True,
        help="Path to an output file to save the generated rectangles in FCML format (e.g., output.txt)."
    )
    parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=10000,
        help="Maximum number of attempts to add or cut line segments per green blob. Higher values "
             "create more complex structures but take longer. Default is 10000."
    )
    parser.add_argument(
        "-L", "--min-line-length-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for `fixed-dimension-size` to set the minimum length for new line segments "
             "during growth. E.g., if thickness is 40 and multiplier is 1.0, min length is 40. Default is 1.0."
    )
    parser.add_argument(
        "-X", "--max-line-length-multiplier",
        type=float,
        default=5.0,
        help="Multiplier for `fixed-dimension-size` to set the maximum length for new line segments "
             "during growth. E.g., if thickness is 40 and multiplier is 5.0, max length is 200. Default is 5.0."
    )
    parser.add_argument(
        "-a", "--initial-segment-attempts",
        type=int,
        default=20,
        help="Number of attempts to place the very first line segment for a blob. Affects initial tree placement. Default is 20."
    )
    parser.add_argument(
        "-A", "--min-blob-area",
        type=int,
        default=50,
        help="Minimum pixel area for a green blob to be processed. Blobs smaller than this are skipped. Default is 50."
    )
    parser.add_argument(
        "-p", "--line-padding",
        type=float,
        default=0.0,
        help="Amount (in pixels) to extend each line segment on both ends after generation. "
             "E.g., 40.0 means 40 pixels added to each end (total +80 length). Default is 0.0."
    )
    parser.add_argument(
        "-b", "--horizontal-bias",
        type=float,
        default=0.5,
        help="Bias towards generating horizontal lines (0.0-1.0). 0.5 is neutral (50/50 chance), "
             ">0.5 prefers horizontal, <0.5 prefers vertical. Default is 0.5."
    )
    parser.add_argument(
        "-c", "--cut-probability",
        type=float,
        default=0.0,
        help="Probability (0.0-1.0) of attempting a line cutting operation instead of a line growth operation "
             "in each iteration. 0.0 disables cutting. Default is 0.0."
    )
    parser.add_argument(
        "-k", "--min-segment-length-after-cut-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for `fixed-dimension-size` to determine the minimum length of *each* segment "
             "resulting from a cut. A line must be at least twice this length to be cut. Default is 1.0."
    )

    args = parser.parse_args()

    # Validate argument ranges to ensure correct behavior.
    if not (0.0 <= args.horizontal_bias <= 1.0):
        parser.error("--horizontal-bias must be between 0.0 and 1.0.")
    if not (0.0 <= args.cut_probability <= 1.0):
        parser.error("--cut-probability must be between 0.0 and 1.0.")

    # Execute the image processing.
    process_image_for_contraption(
        args.image_file,
        args.fixed_dimension_size,
        args.min_spacing,
        args.output_file,
        args.max_iterations,
        args.min_line_length_multiplier,
        args.max_line_length_multiplier,
        args.initial_segment_attempts,
        args.min_blob_area,
        args.line_padding,
        args.horizontal_bias,
        args.cut_probability,
        args.min_segment_length_after_cut_multiplier
    )
