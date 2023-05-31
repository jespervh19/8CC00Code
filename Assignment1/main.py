def read_set_from_file(file: str)->list:
    """
    Read set of intervals from file and return as a list of intervals.
    
    Arguments:
    
    file: the path to the file to be read (string)
    
    Returns:
    A list of intervals read from the file (list).
    Length will be the number of lines
    """
    set = []
    with open(file) as f:
        data = f.read()
        # Split the read file on each line
        data = data.split('\n')
        for line in data:
            line = line.strip('[]')
            intervals = line.split('],[')
            # Find each individual interval, put it in a list and append to the set
            interval = [tuple(map(int, sublist.split(','))) for sublist in intervals]
            set.append(interval)
    return set

def write_score_to_file(outfile: str, similarity_score: float):
    """
    Write similarity score to a file.
    
    Arguments:
    
    outfile: the path to the file to be written (string)
    
    similarity_score: the similarity score to be written to the file (float)
    """
    with open(outfile, 'w') as f:
        f.write(str(similarity_score))

def get_overlap(intervals_1: list, intervals_2: list) -> int:
    """
    Calculate the number of overlapping intervals between two lists of intervals.
    
    Arguments:
    intervals_1: the first list of intervals (list)
    intervals_2: the second list of intervals (list)
    
    Returns:
    The ratio of overlapping intervals when comparing intervals_1 with intervals_2 (float)
    """
    overlap_count = 0
    # Sort the inverval list to be able to do a binary search
    intervals_2 = sorted(intervals_2)
    # Loop over the intervals in the first set
    for interval_1 in intervals_1:
        left = 0
        right = len(intervals_2) - 1
        # Perform the binary search to find out if interval_1 overlaps with any in intervals_2
        # This reduces it to O(n log n)
        while left <= right:
            mid = (left + right) // 2
            interval_2 = intervals_2[mid]
            if interval_1[1] >= interval_2[0] and interval_1[0] <= interval_2[1]:
                overlap_count += 1
                break
            elif interval_1[0] > interval_2[1]:
                left = mid + 1
            else:
                right = mid - 1
    # Divide the found overlap with the maximum length of the two interval lists
    overlap = overlap_count / max(len(intervals_1), len(intervals_2))
    return overlap

def calc_similarity_score(set_1_list: list, set_2_list: list) -> float:
    """
    Calculate the similarity score between two sets of intervals.
    
    Arguments:
    set_1_list: the first list of list of intervals (list)
    set_2_list: the second list of list of intervals (list)
    
    Returns:
    The similarity score between the two sets (float)
    """
    sum = 0
    # Loop over the lines in the sets
    for intervals_1, intervals_2 in zip(set_1_list, set_2_list):
        # Get the overlap when comparing 1 with 2 and vice versa
        overlap_12 = get_overlap(intervals_1, intervals_2)
        overlap_21 = get_overlap(intervals_2, intervals_1)
        sum += (overlap_12 + overlap_21) / 2
    similarity = sum/len(set_1_list)
    return round(similarity, 2)

def similarity(set_1: str, set_2: str, outfile: str) -> float:
    """
    Calculate the similarity score between two sets of intervals read from a file and write the result to a file.
    
    Arguments:
    set_1: the path to the first set of intervals (string)
    set_2: the path to the second set of intervals (string)
    outfile: the path to the file to write the similarity score (string)
    
    Returns:
    The similarity score between the two sets of intervals (float)
    """
    # Read the two sets from file
    set_1_list = read_set_from_file(set_1)
    set_2_list = read_set_from_file(set_2)
    # Calculate the similarity score between the two sets
    similarity_score = calc_similarity_score(set_1_list, set_2_list)
    # Write the similarity score to the outfile
    write_score_to_file(outfile, similarity_score)
    return similarity_score