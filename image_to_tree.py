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
    are the dimensions.
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
    Coordinates are global image coordinates.
    """
    def __init__(self, x1, y1, x2, y2, thickness):
        # Ensure x1 <= x2 and y1 <= y2 for consistent representation
        # Also, ensure start and end points are distinct, or swap if inverted
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.thickness = thickness

        # Determine orientation more robustly by checking actual length
        # Using a tighter tolerance (0.1) to avoid creating effectively zero-length lines
        if abs(self.x1 - self.x2) < 0.1 and abs(self.y1 - self.y2) < 0.1: # It's a point (effectively zero length)
             raise ValueError(f"Line cannot have zero length: ({x1},{y1})-({x2},{y2})")
        elif abs(self.y1 - self.y2) < 0.1: # y-coords are the same (within tolerance)
            self.orientation = 'horizontal'
        elif abs(self.x1 - self.x2) < 0.1: # x-coords are the same (within tolerance)
            self.orientation = 'vertical'
        else:
            raise ValueError(f"Line must be horizontal or vertical, got ({x1},{y1})-({x2},{y2})")

    def __str__(self):
        return (f"Line({self.x1:.2f}, {self.y1:.2f}) to ({self.x2:.2f}, {self.y2:.2f}), "
                f"Thickness={self.thickness:.2f}, Orientation={self.orientation})")

    def to_rectangle(self):
        """Converts the line to a GeneratedRectangle object."""
        if self.orientation == 'horizontal':
            return GeneratedRectangle(
                self.x1,
                self.y1 - self.thickness / 2, # Top-left Y for rectangle
                self.x2 - self.x1,
                self.thickness
            )
        else: # vertical
            return GeneratedRectangle(
                self.x1 - self.thickness / 2, # Top-left X for rectangle
                self.y1,
                self.thickness,
                self.y2 - self.y1
            )

    def intersects_rectangle(self, rect):
        """Checks if this line (as a rectangle) intersects another rectangle."""
        my_rect = self.to_rectangle()
        return (my_rect.x < rect.x + rect.width and
                my_rect.x + my_rect.width > rect.x and
                my_rect.y < rect.y + rect.height and
                my_rect.y + my_rect.height > rect.y)

# --- Helper functions moved to top for proper definition order ---

def _get_random_green_point(local_blob_mask, min_row, min_col):
    """Picks a random green pixel within the blob mask and converts to global coords."""
    green_pixels = np.argwhere(local_blob_mask)
    if not green_pixels.size > 0:
        return None
    
    # Pick a random green pixel (row, col) in local mask coords
    random_idx = random.randint(0, len(green_pixels) - 1)
    local_y, local_x = green_pixels[random_idx]

    # Convert to global coordinates (center of the pixel for smoother line placement)
    global_x = min_col + local_x + 0.5
    global_y = min_row + local_y + 0.5
    return global_x, global_y

def _distance_between_rects(rect1, rect2):
    """
    Calculates the minimum distance between the edges of two axis-aligned rectangles.
    Returns 0 if they overlap or touch.
    """
    # Calculate horizontal gap
    dx = max(0.0, rect1.x - (rect2.x + rect2.width), rect2.x - (rect1.x + rect1.width))
    # Calculate vertical gap
    dy = max(0.0, rect1.y - (rect2.y + rect2.height), rect2.y - (rect1.y + rect1.height))

    # If they overlap in one dimension, the distance in that dimension is 0.
    # The total distance is the maximum of the gaps in x and y directions.
    # This correctly measures the minimum clearance.
    return max(dx, dy)


def _shares_endpoint(line, point_x, point_y, tolerance=0.5): # Reduced tolerance slightly for precision
    """
    Checks if a given point (point_x, point_y) is one of the line's endpoints.
    Uses a tolerance for floating point comparison.
    """
    is_endpoint_1 = (abs(line.x1 - point_x) < tolerance and abs(line.y1 - point_y) < tolerance)
    is_endpoint_2 = (abs(line.x2 - point_x) < tolerance and abs(line.y2 - point_y) < tolerance)
    return is_endpoint_1 or is_endpoint_2


def _check_line_segment_validity(start_x, start_y, end_x, end_y, thickness, local_blob_mask, min_row, min_col):
    """
    Checks if a proposed line segment (with thickness) stays within the green blob mask.
    This is a simplified check that samples points along the line's thickness.
    A more robust check would involve iterating over all pixels covered by the line's rectangle.
    It now returns (bool, reason_string).
    """
    # Convert global coordinates to local blob mask coordinates
    local_x1 = int(start_x - min_col)
    local_y1 = int(start_y - min_row)
    local_x2 = int(end_x - min_col)
    local_y2 = int(end_y - min_row)

    mask_height, mask_width = local_blob_mask.shape

    # Determine line orientation
    is_horizontal = abs(start_y - end_y) < 0.1 # Using 0.1 for consistency with GeneratedLine
    
    # Calculate bounds for thickness
    if is_horizontal:
        center_y_local = int(start_y - min_row)
        min_local_y_thick = int(center_y_local - thickness / 2)
        max_local_y_thick = int(center_y_local + thickness / 2) # Exclusive upper bound
        
        # Clamp thickness bounds to mask dimensions
        min_local_y_thick = max(0, min_local_y_thick)
        max_local_y_thick = min(mask_height, max_local_y_thick)

        # Iterate along the line's length (X-axis)
        for lx in range(min(local_x1, local_x2), max(local_x1, local_x2) + 1):
            # Check all points across the thickness (Y-axis)
            for ly in range(min_local_y_thick, max_local_y_thick):
                # Ensure lx and ly are within mask bounds AND that pixel is green
                if not (0 <= lx < mask_width and 0 <= ly < mask_height and local_blob_mask[ly, lx]):
                    return False, "Outside blob (horizontal)"
        return True, "Within blob (horizontal)"
    else: # Vertical
        center_x_local = int(start_x - min_col)
        min_local_x_thick = int(center_x_local - thickness / 2)
        max_local_x_thick = int(center_x_local + thickness / 2) # Exclusive upper bound
        
        # Clamp thickness bounds to mask dimensions
        min_local_x_thick = max(0, min_local_x_thick)
        max_local_x_thick = min(mask_width, max_local_x_thick)

        # Iterate along the line's length (Y-axis)
        for ly in range(min(local_y1, local_y2), max(local_y1, local_y2) + 1):
            # Check all points across the thickness (X-axis)
            for lx in range(min_local_x_thick, max_local_x_thick):
                # Ensure lx and ly are within mask bounds AND that pixel is green
                if not (0 <= lx < mask_width and 0 <= ly < mask_height and local_blob_mask[ly, lx]):
                    return False, "Outside blob (vertical)"
        return True, "Within blob (vertical)"


def _can_add_line_segment_within_blob(new_line, local_blob_mask, min_row, min_col):
    """
    Checks if a new line segment stays within the green blob mask.
    This function now correctly returns (bool, reason_string) from _check_line_segment_validity.
    """
    return _check_line_segment_validity(new_line.x1, new_line.y1, new_line.x2, new_line.y2, new_line.thickness, local_blob_mask, min_row, min_col)


def _can_add_line_segment(new_line, existing_lines, start_point_x, start_point_y, min_spacing_val, local_blob_mask, min_row, min_col, line_padding):
    """
    Checks if a new line segment is valid to add to the growing tree.

    Rules:
    - Must be within the blob.
    - Must not be redundant (identical or fully contained within an existing non-neighboring line).
    - If it overlaps an existing non-neighboring padded line, reject.
    - If it's too close to any non-neighboring padded line, reject.
    These checks are now performed on the *padded* versions of the rectangles.
    """
    # 1. Check if the proposed line segment is within the green blob mask (unpadded check)
    is_within_blob, blob_reason = _can_add_line_segment_within_blob(new_line, local_blob_mask, min_row, min_col)
    if not is_within_blob:
        return False, blob_reason

    # 2. If this is the very first line being placed, it's valid.
    if not existing_lines:
        return True, "Valid (first segment)"
    
    # Create a padded version of the new_line's rectangle for collision checks
    new_rect_padded = new_line.to_rectangle() # Gets unpadded rect first
    if new_line.orientation == 'horizontal':
        new_rect_padded.x -= line_padding
        new_rect_padded.width += 2 * line_padding
    else: # vertical
        new_rect_padded.y -= line_padding
        new_rect_padded.height += 2 * line_padding

    # Identify true neighbors based on sharing the start_point (current_x, current_y) of the UNPADDED line
    # These are lines that the new_line is connected to by definition.
    neighbors = []
    for existing_line in existing_lines:
        if _shares_endpoint(existing_line, start_point_x, start_point_y, tolerance=0.5):
            neighbors.append(existing_line)
    
    # 3. Apply collision, spacing, and redundancy rules to all existing lines (using their padded versions)
    for existing_line in existing_lines:
        # Get padded version of the existing line's rectangle for collision checks
        existing_rect_padded = existing_line.to_rectangle() # Gets unpadded rect first
        if existing_line.orientation == 'horizontal':
            existing_rect_padded.x -= line_padding
            existing_rect_padded.width += 2 * line_padding
        else: # vertical
            existing_rect_padded.y -= line_padding
            existing_rect_padded.height += 2 * line_padding

        # --- Enhanced Redundancy Check for ALL lines (neighbors and non-neighbors) ---
        # This will catch cases where new_line is identical to, contained in, or contains an existing line
        # regardless of whether it's a "neighbor" or not. The tolerance prevents floating point issues.
        if (new_line.orientation == existing_line.orientation and
            abs(new_line.thickness - existing_line.thickness) < 0.1):
            
            # Check for full containment or near-exact duplicate
            if new_line.orientation == 'horizontal':
                if (new_line.x1 >= existing_line.x1 - 0.1 and new_line.x2 <= existing_line.x2 + 0.1 and
                    abs(new_line.y1 - existing_line.y1) < 0.1):
                    # New line is contained within existing_line (or identical)
                    return False, "Redundant (new line contained in existing collinear line)"
                if (existing_line.x1 >= new_line.x1 - 0.1 and existing_line.x2 <= new_line.x2 + 0.1 and
                    abs(new_line.y1 - existing_line.y1) < 0.1):
                    # Existing_line is contained within new_line (new line is too similar to an already existing line and covers it)
                    return False, "Redundant (new line contains existing collinear line)"
            else: # vertical
                if (new_line.y1 >= existing_line.y1 - 0.1 and new_line.y2 <= existing_line.y2 + 0.1 and
                    abs(new_line.x1 - existing_line.x1) < 0.1):
                    return False, "Redundant (new line contained in existing collinear line)"
                if (existing_line.y1 >= new_line.y1 - 0.1 and existing_line.y2 <= new_line.y2 + 0.1 and
                    abs(new_line.x1 - existing_line.x1) < 0.1):
                    return False, "Redundant (new line contains existing collinear line)"
            
            # Specific check for lines that are very close to duplicates but might not perfectly "contain"
            if (abs(new_line.x1 - existing_line.x1) < 0.1 and abs(new_line.y1 - existing_line.y1) < 0.1 and
                abs(new_line.x2 - existing_line.x2) < 0.1 and abs(new_line.y2 - existing_line.y2) < 0.1):
                return False, "Redundant (near-exact duplicate line)"

        # If the current existing_line is a direct neighbor (parent line for extension),
        # we bypass the strict padded overlap and spacing checks IF it passed the enhanced redundancy checks above.
        # This allows for lines to branch off or extend slightly *without* being seen as general overlaps.
        if existing_line in neighbors:
            continue


        # Rule: If new padded rectangle overlaps existing padded rectangle (that is NOT a neighbor OR is not collinear and thus not caught by redundancy check)
        if (new_rect_padded.x < existing_rect_padded.x + existing_rect_padded.width and
            new_rect_padded.x + new_rect_padded.width > existing_rect_padded.x and
            new_rect_padded.y < existing_rect_padded.y + existing_rect_padded.height and
            new_rect_padded.y + new_rect_padded.height > existing_rect_padded.y):
            return False, "Overlap with non-redundant padded line"

        # Rule: If new padded rectangle is too close to existing padded rectangle (that is NOT a neighbor)
        dist = _distance_between_rects(new_rect_padded, existing_rect_padded)
        if dist < min_spacing_val:
            return False, f"Violates min_spacing ({dist:.2f} < {min_spacing_val:.2f}) with non-redundant padded line"

    # If it passes all checks, it's valid.
    return True, "Valid (connected and spaced with non-redundant lines)"


def _perform_line_cut(generated_lines, min_segment_length_after_cut, local_blob_mask, min_row, min_col, min_spacing, line_padding):
    """
    Attempts to cut an existing line segment into two smaller segments.
    Args:
        generated_lines (list): The current list of GeneratedLine objects.
        min_segment_length_after_cut (float): The minimum allowed length for each new segment created by the cut.
        local_blob_mask: The mask of the current blob.
        min_row, min_col: Bounding box for local_blob_mask.
        min_spacing: Minimum spacing for validity checks of new segments.
        line_padding (float): Amount of padding, needed for _can_add_line_segment.
    Returns:
        bool: True if a cut was successfully performed, False otherwise.
    """
    if not generated_lines:
        return False

    line_to_cut = random.choice(generated_lines)
    
    # Calculate current length
    if line_to_cut.orientation == 'horizontal':
        original_length = abs(line_to_cut.x2 - line_to_cut.x1)
    else: # vertical
        original_length = abs(line_to_cut.y2 - line_to_cut.y1)

    # A line can only be cut if its length is at least twice the minimum segment length after cut
    if original_length < 2 * min_segment_length_after_cut:
        return False # Line is too short to be cut into two valid segments

    # Determine valid range for cut point to ensure both new segments meet min_segment_length_after_cut
    # Cut point position will be a length from the smaller coordinate end of the line
    min_cut_pos_from_start = min_segment_length_after_cut
    max_cut_pos_from_start = original_length - min_segment_length_after_cut

    if max_cut_pos_from_start <= min_cut_pos_from_start:
        return False # Still too short after considering minimum segments length

    cut_pos_along_line = random.uniform(min_cut_pos_from_start, max_cut_pos_from_start)

    new_line1 = None
    new_line2 = None

    if line_to_cut.orientation == 'horizontal':
        # Calculate global x-coordinate of the cut point
        cut_x = min(line_to_cut.x1, line_to_cut.x2) + cut_pos_along_line
        cut_y = line_to_cut.y1 # Y-coordinate remains the same

        # Create two new lines
        new_line1 = GeneratedLine(line_to_cut.x1, line_to_cut.y1, cut_x, cut_y, line_to_cut.thickness)
        new_line2 = GeneratedLine(cut_x, cut_y, line_to_cut.x2, line_to_cut.y2, line_to_cut.thickness)
    else: # vertical
        # Calculate global y-coordinate of the cut point
        cut_y = min(line_to_cut.y1, line_to_cut.y2) + cut_pos_along_line
        cut_x = line_to_cut.x1 # X-coordinate remains the same

        # Create two new lines
        new_line1 = GeneratedLine(line_to_cut.x1, line_to_cut.y1, cut_x, cut_y, line_to_cut.thickness)
        new_line2 = GeneratedLine(cut_x, cut_y, line_to_cut.x2, line_to_cut.y2, line_to_cut.thickness)
    
    # Before adding, check if these new lines would violate any rules with *other* existing lines
    # (they are inherently valid with each other at the cut point, and replace original)
    
    # Temporarily remove line_to_cut for validation against others
    temp_existing_lines = [l for l in generated_lines if l != line_to_cut]

    # When checking new_line1 and new_line2, their 'neighbors' would be each other at the cut point,
    # but more importantly, other parts of the original tree.
    # We pass the original line_to_cut as the 'start_point' for new_line1 and new_line2's
    # connectivity check if they are directly replacing it and connecting to its original endpoints.
    # This logic is complex for cuts, as they are replacing. The validation here needs to ensure
    # the new segments don't immediately conflict with *other* non-connected parts of the tree.
    # The existing _can_add_line_segment, when given `temp_existing_lines`, should work as intended.
    is_valid1, _ = _can_add_line_segment(new_line1, temp_existing_lines, new_line1.x1, new_line1.y1, min_spacing, local_blob_mask, min_row, min_col, line_padding)
    is_valid2, _ = _can_add_line_segment(new_line2, temp_existing_lines, new_line2.x2, new_line2.y2, min_spacing, local_blob_mask, min_row, min_col, line_padding) # Use x2,y2 as "start" for validation against others

    if is_valid1 and is_valid2:
        generated_lines.remove(line_to_cut)
        generated_lines.append(new_line1)
        generated_lines.append(new_line2)
        # print(f"  Cut line: {line_to_cut} into {new_line1} and {new_line2}")
        return True
    
    return False


def process_green_blob(blob_id, region, fixed_dimension_size, min_spacing, max_iterations, min_line_length_multiplier, max_line_length_multiplier, initial_segment_attempts, line_padding, horizontal_bias, cut_probability, min_segment_length_after_cut_multiplier):
    """
    This function processes a single contiguous green region (blob) using the
    specified random walk algorithm to generate a tree-like structure of thick lines.

    Args:
        blob_id (int): A unique identifier for this green blob.
        region (skimage.measure._regionprops.RegionProperties): The properties object for the blob.
        fixed_dimension_size (float): The desired thickness for the lines.
        min_spacing (float): The minimum spacing to enforce between non-connected orthogonal rectangles.
        max_iterations (int): Maximum attempts to add line segments.
        min_line_length_multiplier (float): Multiplier for fixed_dimension_size for min line length.
        max_line_length_multiplier (float): Multiplier for fixed_dimension_size for max line length.
        initial_segment_attempts (int): Number of attempts to place the first segment.
        line_padding (float): Amount to extend each line segment on both ends.
        horizontal_bias (float): Bias towards horizontal lines (0.0-1.0, 0.5 is neutral).
        cut_probability (float): Probability (0.0-1.0) of attempting a line cut instead of a grow.
        min_segment_length_after_cut_multiplier (float): Multiplier for fixed_dimension_size for min segment length after cut.
    Returns:
        list[GeneratedRectangle]: A list of rectangles generated for this blob.
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


    generated_lines = []
    # Mask of the blob relative to its bounding box (True for green pixels)
    local_blob_mask = region.image

    start_point = _get_random_green_point(local_blob_mask, min_row, min_col)
    if start_point is None:
        print("  Could not find a valid starting point within the green blob. Skipping.")
        return []

    initial_x, initial_y = start_point
    line_thickness = fixed_dimension_size
    min_line_length = line_thickness * min_line_length_multiplier
    max_line_length = line_thickness * max_line_length_multiplier
    min_segment_length_after_cut = line_thickness * min_segment_length_after_cut_multiplier


    print(f"  Starting growth from ({initial_x:.2f}, {initial_y:.2f}) with max iterations {max_iterations}...")

    # Create the very first segment
    initial_segment_added = False
    for _ in range(initial_segment_attempts): # Use initial_segment_attempts
        try:
            # Apply bias when choosing initial orientation
            if random.random() < horizontal_bias:
                current_orientation = 'horizontal'
            else:
                current_orientation = 'vertical'
            
            direction_multiplier = random.choice([1, -1])
            length = random.uniform(min_line_length, max_line_length)

            if current_orientation == 'horizontal':
                end_x = initial_x + length * direction_multiplier
                initial_line = GeneratedLine(initial_x, initial_y, end_x, initial_y, line_thickness)
            else: # Vertical
                end_y = initial_y + length * direction_multiplier
                initial_line = GeneratedLine(initial_x, initial_y, initial_x, end_y, line_thickness)
            
            # For the very first line, its starting point is initial_x, initial_y.
            # There are no existing_lines. parent_line_obj is None.
            # Pass line_padding to _can_add_line_segment for the initial check
            is_valid, _ = _can_add_line_segment(initial_line, [], initial_x, initial_y, min_spacing, local_blob_mask, min_row, min_col, line_padding)
            if is_valid:
                generated_lines.append(initial_line)
                initial_segment_added = True
                print(f"  Initial segment added: {initial_line}")
                break
        except ValueError as e:
            pass # Ignore invalid line constructions (e.g., zero length)
    
    if not initial_segment_added:
        print("  Could not generate a valid initial segment. Skipping blob.")
        return []

    # Main growth loop
    start_time_blob = time.time()
    last_print_time = start_time_blob
    for i in range(max_iterations): # Use max_iterations
        # Print progress every second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            progress_percent = (i / max_iterations) * 100
            print(f"  Blob #{blob_id} Progress: Iteration {i}/{max_iterations} ({progress_percent:.2f}%)")
            last_print_time = current_time

        # Decide whether to grow a new line or cut an existing one
        if generated_lines and random.random() < cut_probability:
            # Attempt a cut operation
            _perform_line_cut(generated_lines, min_segment_length_after_cut, local_blob_mask, min_row, min_col, min_spacing, line_padding) # Pass line_padding
        else:
            # Attempt a grow operation
            # Pick a random point in our tree (an endpoint of an existing line)
            parent_line_from_tree = random.choice(generated_lines) # The line we're conceptually extending from
            start_point_candidates = [(parent_line_from_tree.x1, parent_line_from_tree.y1),
                                  (parent_line_from_tree.x2, parent_line_from_tree.y2)]
            current_x, current_y = random.choice(start_point_candidates) # This is the "start_point" for the new line.

            # Apply bias when picking random direction
            if random.random() < horizontal_bias: # Prefer horizontal direction
                chosen_orientation = 'horizontal'
            else: # Prefer vertical direction
                chosen_orientation = 'vertical'

            # Based on chosen orientation, pick a direction multiplier
            if chosen_orientation == 'horizontal':
                dir_x_mult = random.choice([1, -1])
                dir_y_mult = 0
            else: # vertical
                dir_x_mult = 0
                dir_y_mult = random.choice([1, -1])

            # Pick a random length
            length = random.uniform(min_line_length, max_line_length)

            # Calculate proposed end point
            potential_end_x = current_x + length * dir_x_mult
            potential_end_y = current_y + length * dir_y_mult

            # Create new candidate segment
            try:
                new_line_candidate = GeneratedLine(current_x, current_y, potential_end_x, potential_end_y, line_thickness)
            except ValueError: # Catch cases where line is invalid (e.g., zero length after rounding/tiny movement)
                continue # Try next iteration

            # --- Validation for new segment ---
            # Pass the current_x, current_y as the start_point for neighbor identification
            is_valid, reason = _can_add_line_segment(new_line_candidate, generated_lines, current_x, current_y, min_spacing, local_blob_mask, min_row, min_col, line_padding)
            
            if is_valid:
                generated_lines.append(new_line_candidate)
            # else:
                # print(f"  Rejected (iteration {i + 1}): {reason} for {new_line_candidate}")

    time_taken_blob = time.time() - start_time_blob
    print(f"  Blob #{blob_id} processing finished. Time taken: {time_taken_blob:.2f} seconds.")


    print(f"  Growth completed. Generated {len(generated_lines)} raw line segments.")

    # --- Merge colinear and connected lines ---
    print(f"  Attempting to merge {len(generated_lines)} generated lines...")
    merged_lines = merge_collinear_lines(generated_lines)
    print(f"  After merging, {len(merged_lines)} lines remain.")

    final_rectangles = []
    for line in merged_lines: # Use merged_lines for final output
        rect = line.to_rectangle() # This gives the original top-left rect

        # Apply padding based on line orientation
        # The padding extends the line, so adjust top-left corner and dimension
        if line.orientation == 'horizontal':
            rect.x -= line_padding
            rect.width += 2 * line_padding
        else: # vertical
            rect.y -= line_padding
            rect.height += 2 * line_padding
            
        # Calculate center coordinates and add to final list
        center_x = rect.x + rect.width / 2
        center_y = rect.y + rect.height / 2
        final_rectangles.append(GeneratedRectangle(center_x, center_y, rect.width, rect.height))


    print(f"  Final generated {len(final_rectangles)} rectangles for this blob.")
    return final_rectangles

# Merging function (re-enabled and checked)
def merge_collinear_lines(lines):
    """
    Merges collinear and connected lines into longer segments.
    This implementation iterates and merges, reducing the total number of lines.
    It processes horizontal lines first, then vertical lines.
    """
    if not lines:
        return []

    # Separate horizontal and vertical lines
    horizontal_lines = [line for line in lines if line.orientation == 'horizontal']
    vertical_lines = [line for line in lines if line.orientation == 'vertical']

    merged_h_lines = []
    # Sort by y, then x1 to make merging easier
    horizontal_lines.sort(key=lambda l: (l.y1, l.x1))
    
    # Merge horizontal lines
    if horizontal_lines:
        current_merged = horizontal_lines[0]
        for i in range(1, len(horizontal_lines)):
            line = horizontal_lines[i]
            
            # Check if lines are on the same y-coordinate (within tolerance)
            # and if their thicknesses are the same (within tolerance)
            is_collinear_and_same_thickness = (
                abs(current_merged.y1 - line.y1) < 0.1 and
                abs(current_merged.thickness - line.thickness) < 0.1
            )
            
            # Check for direct contiguity / touching or overlapping for merging
            # Allows merging if they overlap or touch with some tolerance
            overlaps_or_touches = (line.x1 <= current_merged.x2 + 0.1 and line.x2 >= current_merged.x1 - 0.1)

            if is_collinear_and_same_thickness and overlaps_or_touches:
                # Merge: extend current_merged to cover the new line's extent
                current_merged.x1 = min(current_merged.x1, line.x1)
                current_merged.x2 = max(current_merged.x2, line.x2)
            else:
                merged_h_lines.append(current_merged)
                current_merged = line
        merged_h_lines.append(current_merged) # Add the last merged line

    merged_v_lines = []
    # Sort by x, then y1 to make merging easier
    vertical_lines.sort(key=lambda l: (l.x1, l.y1))

    # Merge vertical lines
    if vertical_lines:
        current_merged = vertical_lines[0]
        for i in range(1, len(vertical_lines)):
            line = vertical_lines[i]
            
            # Check if lines are on the same x-coordinate (within tolerance)
            # and if their thicknesses are the same (within tolerance)
            is_collinear_and_same_thickness = (
                abs(current_merged.x1 - line.x1) < 0.1 and
                abs(current_merged.thickness - line.thickness) < 0.1
            )
            
            # Check for direct contiguity / touching or overlapping for merging
            # Allows merging if they overlap or touch with some tolerance
            overlaps_or_touches = (line.y1 <= current_merged.y2 + 0.1 and line.y2 >= current_merged.y1 - 0.1)
            
            if is_collinear_and_same_thickness and overlaps_or_touches:
                # Merge: extend current_merged to cover the new line's extent
                current_merged.y1 = min(current_merged.y1, line.y1)
                current_merged.y2 = max(current_merged.y2, line.y2)
            else:
                merged_v_lines.append(current_merged)
                current_merged = line
        merged_v_lines.append(current_merged) # Add the last merged line

    return merged_h_lines + merged_v_lines


def process_image_for_contraption(image_path, fixed_dimension_size, min_spacing, output_file, max_iterations, min_line_length_multiplier, max_line_length_multiplier, initial_segment_attempts, min_blob_area, line_padding, horizontal_bias, cut_probability, min_segment_length_after_cut_multiplier):
    """
    This function processes an image to generate Fantastic Contraption
    level decorations by identifying and processing contiguous green regions.
    It now outputs the generated rectangles to a specified file in FCML format.

    Args:
        image_path (str): The file path to the input image.
        fixed_dimension_size (float): The desired thickness for the lines.
        min_spacing (float): The minimum spacing to enforce between non-connected rectangles.
        output_file (str): The path to the output file for FCML data.
        max_iterations (int): Maximum attempts to add line segments.
        min_line_length_multiplier (float): Multiplier for fixed_dimension_size for min line length.
        max_line_length_multiplier (float): Multiplier for fixed_dimension_size for max line length.
        initial_segment_attempts (int): Number of attempts to place the first segment.
        min_blob_area (int): Minimum pixel area for a blob to be processed.
        line_padding (float): Amount to extend each line segment on both ends.
        horizontal_bias (float): Bias towards horizontal lines (0.0-1.0, 0.5 is neutral).
        cut_probability (float): Probability (0.0-1.0) of attempting a line cut instead of a grow.
        min_segment_length_after_cut_multiplier (float): Multiplier for fixed_dimension_size for min segment length after cut.
    """
    print(f"Received image for processing: {image_path}")
    all_generated_rectangles = [] # Collect all rectangles here

    overall_start_time = time.time() # Start overall timer

    try:
        with Image.open(image_path) as img:
            print(f"Image successfully loaded. Format: {img.format}, Size: {img.size}")

            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("Image converted to RGB mode.")

            img_np = np.array(img)

            lower_green = np.array([0, 100, 0])
            upper_green = np.array([100, 255, 100])

            green_mask = np.all((img_np >= lower_green) & (img_np <= upper_green), axis=-1)
            labeled_mask = label(green_mask)
            regions = regionprops(labeled_mask)

            print(f"\nFound {len(regions)} contiguous green regions.")

            for i, region in enumerate(regions):
                if region.area > min_blob_area: # Use min_blob_area parameter
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
                    print(f"\n--- Skipping small blob #{region.label} with area {region.area} ---")

            if not regions:
                print("No green regions found within the specified color range.")

        # Write rectangles to output file
        print(f"\nWriting {len(all_generated_rectangles)} rectangles to {output_file}...")
        with open(output_file, 'w') as f:
            for rect in all_generated_rectangles:
                # FCML format: SR x y width height (where x,y are CENTER coordinates)
                f.write(f"SR {rect.x:.2f} {rect.y:.2f} {rect.width:.2f} {rect.height:.2f}\n")
        print("Finished writing FCML file.")


    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred while opening or processing the image: {e}")
    
    overall_time_taken = time.time() - overall_start_time # Calculate overall time
    print(f"\nOverall processing completed. Total time taken: {overall_time_taken:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Fantastic Contraption level decorations from an image."
    )
    parser.add_argument(
        "-i", "--image-file",
        type=str,
        required=True,
        help="The path to the input image file (e.g., PNG, JPEG)."
    )
    parser.add_argument(
        "-t", "--fixed-dimension-size",
        type=float,
        default=40.0,
        help="The fixed thickness for the generated lines/rectangles. Default is 40."
    )
    parser.add_argument(
        "-s", "--min-spacing",
        type=float,
        default=40.0,
        help="The minimum spacing to enforce between non-connected orthogonal rectangles. Default is 40."
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
        help="Maximum number of iterations to attempt adding line segments per blob. Default is 10000."
    )
    parser.add_argument(
        "-L", "--min-line-length-multiplier", # Shorthand added
        type=float,
        default=1.0,
        help="Multiplier for fixed-dimension-size to determine the minimum line segment length for new growth. Default is 1.0."
    )
    parser.add_argument(
        "-X", "--max-line-length-multiplier", # Shorthand added
        type=float,
        default=5.0,
        help="Multiplier for fixed-dimension-size to determine the maximum line segment length for new growth. Default is 5.0."
    )
    parser.add_argument(
        "-a", "--initial-segment-attempts", # Shorthand added
        type=int,
        default=20,
        help="Number of attempts to place the very first line segment for a blob. Default is 20."
    )
    parser.add_argument(
        "-A", "--min-blob-area", # Shorthand added
        type=int,
        default=50,
        help="Minimum pixel area for a green blob to be processed. Smaller blobs are skipped. Default is 50."
    )
    parser.add_argument(
        "-p", "--line-padding",
        type=float,
        default=0.0, # Default to 0, so it doesn't pad unless specified
        help="Amount to extend each line segment on both ends (e.g., 40.0 means 40 pixels added to each end). Default is 0.0."
    )
    parser.add_argument(
        "-b", "--horizontal-bias",
        type=float,
        default=0.5, # Default to 0.5 (neutral)
        help="Bias towards horizontal lines (0.0-1.0, 0.5 is neutral, >0.5 prefers horizontal, <0.5 prefers vertical). Default is 0.5."
    )
    parser.add_argument(
        "-c", "--cut-probability",
        type=float,
        default=0.0, # Default to 0.0 (no cutting)
        help="Probability (0.0-1.0) of attempting a line cutting operation instead of a grow operation. Default is 0.0."
    )
    parser.add_argument(
        "-k", "--min-segment-length-after-cut-multiplier", # New parameter with shorthand
        type=float,
        default=1.0, # Default to 1.0 * thickness, meaning each new segment is at least one thickness long
        help="Multiplier for fixed-dimension-size to determine the minimum length of *each* segment resulting from a cut. Default is 1.0."
    )


    args = parser.parse_args()

    # Validate horizontal_bias and cut_probability ranges
    if not (0.0 <= args.horizontal_bias <= 1.0):
        parser.error("--horizontal-bias must be between 0.0 and 1.0.")
    if not (0.0 <= args.cut_probability <= 1.0):
        parser.error("--cut-probability must be between 0.0 and 1.0.")


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
