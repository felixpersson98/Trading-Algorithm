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
# If the algorithm predicts a change within the FUTURE_OFFSET of less than 0.6% The algorithm does give the prediction. 
# as a recommendation
REQUIRED_CONFIDENCE = 0.006

# For the algorithm to work correctly TIME_INTERVAL needs to be equal to FUTURE_OFFSET, it is how often the algorithm
# trades.
TIME_INTERVAL = FUTURE_OFFSET
# INTERVAL_COUNT * TIME_INTERVAL equals the amount of historical data you will trade on.
# i.e INTERVAL_COUNT = 64. FUTURE_OFFSET = 5 => Total amount of days that you trade for is 320 trading days
INTERVAL_COUNT = 64

# The variables below are created so that the user can see how far the algorithm is in the process of trading. 
PROGRESS_BAR_MAX = 100
progress = 1
progress_bar = '='

# total_return is created to display for the user how much return trading on the chosen time period resulted in.
total_return = 1

# Different variables to decide whether the entire trading process was worthy
# recommendations is created to cound the win-rate/hit-rate of the algorithm. 
# If a trade was profitable a "1" is being added whereas a "-1" gets added whenever a trade was wrongly made
recommendations = []
algorithm_growth = []

data = pd.read_csv('PUT PATH TO CSV FILE HERE', delimiter=';').values
data_EURUSD = [x[1] for x in data]
# Can be interesting do use an index to compare to but not necessary.
data_sp500 = [x[2] for x in data]

# Might be necessary to reverse the data to get the dates in the right order.
# data_EURUSD.reverse()
# data_sp500.reverse()

# np_data corresponds to data but is of the type numpy array.
np_data = np.array(data_EURUSD)
np_data_sp500 = np.array(data_sp500)


# x-axis for target pattern
x_normal = np.arange(0, PATTERN_LENGTH)
# x-axis for pattern matches (same as normal but includes FUTURE_OFFSET)
x_with_future = np.arange(0, PATTERN_LENGTH + FUTURE_OFFSET)


# Defines a function that searches through historical data for matching pattern
# Required input so that future patterns are not being stored
def examine_current_pattern(input_data_currencies):
    current_data_length = len(input_data_currencies)
    
    # create a 4-tuple for the best matches array and their corresponding values.
    top_matches = [([], 9999, -1, -1)]  # (pat, diff, future_relative_cp, start)
    
    # Now a historical future value is being stored to calculate the return.
    future_point = input_data_currencies[-1]

    # loads the target pattern and declares two variable containing the start and end value.
    target_pattern = input_data_currencies[-(PATTERN_LENGTH + FUTURE_OFFSET):-FUTURE_OFFSET]
    first_point, last_point = target_pattern[0], target_pattern[-1]

    # Converts the np-array to a change in percent array.
    future_relative_tp = future_point / target_pattern[0] - 1
    target_pattern = target_pattern / target_pattern[0] - 1

    # Loops through all the input data to search for similar patterns.
    for start in range(0, current_data_length - PATTERN_LENGTH * 2 + 1):
        
        # Get data_EURUSD with same size as target and some extra for prediction, current_pattern = historical_pattern
        current_pattern = input_data_currencies[start: start + PATTERN_LENGTH + FUTURE_OFFSET]

        # Get the future_relative_cp value of this current pattern.
        future_relative_cp = current_pattern[-1] / current_pattern[0] - 1

        # Transform current pattern to be relative to first value.
        current_pattern = current_pattern / current_pattern[0]-1

        # Calculate the difference between target pattern, lower is more similar.
        # Current_pattern includes future_relative_cp offset as well so we need to trim it (while
        # comparing) to the right size to compare with target_pattern.
        diff = sum(np.absolute(current_pattern[:PATTERN_LENGTH] - target_pattern)) / PATTERN_LENGTH

        # If difference is lower than previously worst saved, add it to the list.
        if diff < top_matches[-1][1]:
            # Store the candidate in a tuple to group with diff and future_relative_cp.
            new_candidate = (current_pattern, diff, future_relative_cp, start)
            # Add to list of top matches.
            top_matches.append(new_candidate)

            # Sort top matches so that best is first and worst last (sort by diff).
            top_matches.sort(key=lambda match: match[1])

            # Truncate list if it exceeds maximum length.
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

    # Makes the prediction, takes an average of the outcome from the best matching patterns.
    prediction = sum(map(lambda m: m[2], top_matches)) / len(top_matches)
    relative_change_percent = abs((future_point - last_point)) / last_point
    predicted_relative_change = abs((1+prediction)*first_point - last_point) / last_point
    
    # Checks if the prediction was right or wrong.
    actual_outcome = 'RISE' if future_relative_tp > target_pattern[PATTERN_LENGTH - 1] else 'FALL'
    predicted_outcome = 'RISE' if prediction > target_pattern[PATTERN_LENGTH - 1] else 'FALL'
    
    # In order to plot each trading decision uncomment the commented lines below.
    
    # for match in top_matches:
    #     pat = match[0]
    #     plt.plot(x_normal, pat[:30], color='gray')
    # #
    # # # Plot real outcome in orange
    # plt.scatter(x_with_future[-1], future_relative_tp, color='navy')
    # # Plot the pattern we were looking for in blue
    # plt.plot(x_normal, target_pattern, color='navy', linewidth=3)
    #
    # # Plot predicton in red
    # if predicted_relative_change < 0.006:
    #     plt.scatter(x_with_future[-1], prediction, color='gray')
    # else:
    #     if predicted_outcome == actual_outcome:
    #         plt.scatter(x_with_future[-1], prediction, color='green')
    #     else:
    #         plt.scatter(x_with_future[-1], prediction, color='red')
    # # Plot real outcome in orange
    # # Plot the pattern we were looking for in blue
    # plt.show()
    
    return actual_outcome, predicted_outcome, relative_change_percent, predicted_relative_change


# The two arrays below will be the x-axis in the plot further on. 
# In order to plot the growth of the algorithm next to the chosen index you need to change the steps on the second
# array, since the algorithm growth area will contain less datapoints if not trading daily.
x_tot = np.arange(0, INTERVAL_COUNT)
x_tot_future_offset_days = np.arange(0, INTERVAL_COUNT, 1/FUTURE_OFFSET)

# Convert the chosen index to percent change
sp500 = np.array(np_data_sp500[-INTERVAL_COUNT*TIME_INTERVAL:])
sp500 /= sp500[0]

# A for-loop for trading over the entire period
for interval in reversed(range(1, INTERVAL_COUNT + 1)):
    interval_data_currencies = np_data[:-interval * TIME_INTERVAL]
    real, predicted, relative_change, predicted_change = examine_current_pattern(interval_data_currencies)

    # If the predicted_change is lower than the required confidence, no trade has been made => no return.
    if predicted_change < REQUIRED_CONFIDENCE:
        algorithm_growth.append(total_return)
        continue
    total_return *= 1 + (relative_change if real == predicted else -relative_change)

    # Append the return in this very trade and append recommendation to determine the hit-rate
    algorithm_growth.append(total_return)
    recommendations.append(real == predicted)

    # Print the progress
    progress = (INTERVAL_COUNT - (interval-1))/INTERVAL_COUNT
    current_bar_width = int(86*progress)*progress_bar + '>'
    print('\r[{: <87}] {:.0%}'.format(current_bar_width, progress), end='')

# The performance of the algorithm is being plotted next to the relative change in the same time period
plt.title('Performance of Trading Strategy')
plt.plot(x_tot_future_offset_days, sp500, color='orange', linewidth=2)
plt.plot(x_tot, algorithm_growth, color='navy', linewidth=2)
plt.show()
print('\nThe hit rate is {:.2%}'.format(sum(recommendations)/len(recommendations)) + '\n')
print('The total return/loss is {:.2%}'.format(total_return-1))
recommendations = list(map(lambda r: 1 if r else -1, recommendations))
print(len(recommendations))
plt.plot(np.arange(len(recommendations)), recommendations)
plt.show()
