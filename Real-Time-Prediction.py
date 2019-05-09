import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Constants
# PATTERN_LENGTH corresponds to the length of the pattern to be examined and also the historical matching patterns.
PATTERN_LENGTH = 30

# FUTURE_OFFSET represents for how many days into the future you would like to make the prediction.
FUTURE_OFFSET = 5

# TOP_MATCH_HARD_MAX represents the amount of historical patterns to be taken into account when making the prediction.
TOP_MATCH_HARD_MAX = 5

# SOFT_MAX corresponds to the maximum amount of patterns stored in the matching pattern array.
TOP_MATCH_SOFT_MAX = 15

# REQUIRED_CONFIDENCE is used to reduce the amount of wrong predictions.
# If the algorithm predicts a change within the FUTURE_OFFSET of less than 0.6% The algorithm does give the prediction 
# as a recommendation
REQUIRED_CONFIDENCE = 0.006

# Reading the data from a CSV-file, one you have to download yourself.
data = pd.read_csv('PUT PATH TO CSV FILE HERE', delimiter=';').values
data_EURUSD = [x[1] for x in data]

# Might be necessary to reverse the data to get the dates in the right order.
# data_EURUSD.reverse()

# np_data corresponds to data but is of the type numpy array.
np_data = np.array(data_EURUSD)
data_length = len(np_data)

# x-axis for target pattern.
x_normal = np.arange(0, PATTERN_LENGTH)
# x-axis for pattern matches (same as normal but includes FUTURE_OFFSET).
x_with_future = np.arange(0, PATTERN_LENGTH + FUTURE_OFFSET)


def examine_target_pattern():
    # create a 4-tuple for the best matches array and their corresponding values.
    top_matches = [([], 9999, -1, -1)]  # (pattern, diff, future_value_current_pattern, start)

    # loads the target pattern and declares two variable containing the start and end value.
    target_pattern = np_data[-PATTERN_LENGTH:]
    first_point, last_point = target_pattern[0], target_pattern[-1]

    # Converts the np-array to a change in percent array.
    target_pattern = target_pattern / target_pattern[0] - 1

    # Loops through all the data to search for similar patterns.
    for start in range(0, data_length - PATTERN_LENGTH * 2 + 1):
        # Get data_EURUSD with same size as target and some extra for the prediction, cp stands for current pattern.
        # In other words, the historical pattern to be compared
        current_pattern = np_data[start: start + PATTERN_LENGTH + FUTURE_OFFSET]

        # Get the relative future value from historical pattern value of this current pattern.
        future_relative_cp = current_pattern[-1] / current_pattern[0] - 1

        # Transform current pattern to be relative to first value
        current_pattern = current_pattern / current_pattern[0]-1

        # Calculate the difference between target pattern, lower is more similar.
        # current_pattern includes future_relative_cp offset as well so we need to trim it (while
        # comparing) to the right size to compare with target_pattern.
        diff = np.sum(np.absolute(current_pattern[:PATTERN_LENGTH] - target_pattern)) / PATTERN_LENGTH

        # If difference is lower than previously worst saved, add it to the list.
        if diff < top_matches[-1][1]:
            # Store the candidate in a tuple to group with diff and future_relative_cpn.
            new_candidate = (current_pattern, diff, future_relative_cp, start)

            # Add to list of top matches
            top_matches.append(new_candidate)

            # Sort top matches so that best is first and worst last (sort by diff).
            top_matches.sort(key=lambda match: match[1])

            # Truncate list if it exceeds maximum length
            if len(top_matches) > TOP_MATCH_SOFT_MAX:
                top_matches.pop()

    # The while-loop below makes sure that no patterns from the same time period are being stored and make sure
    # that the best one is the only one to be stored. That is why both HARD_MAX and SOFT_MAX are being used.
    index = 0
    while index < len(top_matches):
        current_start = top_matches[index][3]
        ti = top_matches[:index+1]
        tf = list(filter(lambda match: abs(match[3]-current_start) > 10, top_matches[index:]))
        top_matches = [*ti, *tf]
        index += 1
    # Pops the worst (HARD_MAX-SOFT_MAX) patterns.
    top_matches = top_matches[:TOP_MATCH_HARD_MAX]
    plot_color = 'g'
    prediction = sum(map(lambda m: m[2], top_matches)) / len(top_matches)
    predicted_relative_change = abs((1 + prediction) * first_point - last_point) / last_point
    if predicted_relative_change < REQUIRED_CONFIDENCE:
        plot_color = 'gray'

    for match in top_matches:
        pat = match[0]
        plt.plot(x_with_future, pat, color='gray')

    # Plot the pattern we were looking for in navy blue.
    plt.plot(x_normal, target_pattern, color='navy', linewidth=3)

    # Plot predicton in green or gray depending on confidence.
    plt.scatter(x_with_future[-1], prediction, color=plot_color)
    if prediction < target_pattern[-1]:
        print('Predicted relative change:', -predicted_relative_change)
    else:
        print('Predicted relative change:', predicted_relative_change)
    plt.show()


examine_target_pattern()

