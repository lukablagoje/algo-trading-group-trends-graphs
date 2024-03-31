import pandas as pd
import pandas_datareader as pdr 
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import networkx as nx
import datetime
from scipy.stats import pearsonr

def get_data(tickers, start_date, end_date):
    """
    Downloads and processes stock data for a given list of tickers within a specified date range.

    This function fetches the Adjusted Close prices and calculates the daily returns for each ticker
    provided. It ensures that the data covers the exact date range requested and combines the results into two
    dataframes: one for prices and one for daily returns, with missing values cleaned.

    Parameters:
    tickers (list): A list of stock ticker symbols.
    start_date (str): The start date for the data retrieval, in YYYY-MM-DD format.
    end_date (str): The end date for the data retrieval, in YYYY-MM-DD format.

    Returns:
    tuple: Contains two pandas DataFrames. The first DataFrame contains the Adjusted Close prices for each ticker,
    and the second contains the daily returns. If no data is available, empty lists are returned.
    """
    # Initialize variables to hold the number of tickers, and lists for price and daily returns dataframes
    price_list_df = []
    daily_returns_list_df = []
    
    # Convert start and end dates from strings to pandas Timestamps for comparison
    
    start_date_datetime = pd.Timestamp(start_date)
    end_date_datetime = pd.Timestamp(end_date)
    
    # Iterate over each ticker in the list
    for index, ticker in enumerate(tickers):
        print(index, len(tickers))  # Debug print: current index and total tickers
        sp = yf.download(ticker, start=start_date, end=end_date)  # Download stock data
        
        # Check if the download returned data
        if len(sp) > 0:
            sp_mod = sp[["Adj Close"]].copy()  # Copy the Adjusted Close prices
            # Calculate daily returns and add as a new column
            sp_mod.loc[:, "Daily Return"] = sp_mod["Adj Close"] / sp_mod["Adj Close"].shift(1) - 1
            sp_mod[ticker] = sp_mod['Adj Close']  # Duplicate 'Adj Close' column for easier access
            
            # Determine the first and last date of the downloaded data
            min_date = sp_mod[ticker].index[0]
            max_date = sp_mod[ticker].index[-1]
            print(min_date, start_date_datetime)  # Debug print: first date of data vs. requested start
            print(max_date, end_date_datetime)  # Debug print: last date of data vs. requested end
            
            # Check if the data covers the desired date range exactly
            if (min_date == start_date_datetime + datetime.timedelta(days=1)) and (max_date == end_date_datetime - datetime.timedelta(days=1)):
                price_list_df.append(sp_mod[[ticker]].copy())  # Add price data to list
                sp_mod[ticker] = sp_mod['Daily Return']  # Replace price data with daily returns
                daily_returns_list_df.append(sp_mod[ticker])  # Add daily returns data to list
    
    # Check if we have collected any data
    if len(price_list_df) > 0:
        # Concatenate all price dataframes and drop missing values
        price_df = pd.concat(price_list_df, axis=1).dropna()
        # Concatenate all daily returns dataframes and drop missing values
        daily_returns_df = pd.concat(daily_returns_list_df, axis=1).dropna()
    else:
        price_df = []
        daily_returns_df = []
    
    return price_df, daily_returns_df


def find_correlation_groups(time_window_splits, price_df, link_threshold, frequency_threshold, draw):
    """
    Identifies groups of assets with strong correlations within specified time windows.
    
    Parameters:
    - time_window_splits: Indices to split the price DataFrame into time windows.
    - price_df: DataFrame containing price data for different assets.
    - link_threshold: Correlation coefficient threshold for linking two assets.
    - frequency_threshold: Threshold for the frequency of a link appearing across time windows to be considered significant.
    - draw: Boolean flag indicating whether to draw the network graph.
    
    Returns:
    - List of sets, each set contains tickers of assets that are strongly correlated within the time windows.
    """

    # Calculate correlation matrices for each time window
    corr_df_dict = {}
    for i in range(1, len(time_window_splits)):
        # Compute correlation matrix for each time window and store it
        corr_df_dict[i] = price_df.iloc[time_window_splits[i-1]:time_window_splits[i], :].corr()

    # Prepare asset ticker list from the columns of the first correlation matrix
    asset_ticker_list = list(corr_df_dict[1].columns)
    
    # Initialize dictionary to hold correlation pairs for each time window
    correlation_pairs_dict = {}
    for time_window_index in corr_df_dict.keys():
        correlation_pairs_dict[time_window_index] = {}
        # Iterate over all possible asset pairs
        for ticker_1 in asset_ticker_list:
            for ticker_2 in asset_ticker_list:
                if ticker_1 != ticker_2:  # Avoid self-correlation
                    # Store the correlation coefficient between each pair
                    correlation_pairs_dict[time_window_index][(ticker_1, ticker_2)] = corr_df_dict[time_window_index].loc[ticker_1, ticker_2]

    # Identify strongly correlated pairs and build a network for each time window
    network_dict = {}
    for time_window_index in correlation_pairs_dict.keys():
        G = nx.Graph()  # Initialize a new graph for the time window
        for pair, correlation in correlation_pairs_dict[time_window_index].items():
            if correlation >= link_threshold:  # Check if the correlation exceeds the threshold
                G.add_edge(pair[0], pair[1])  # Add an edge for strongly correlated pairs
        network_dict[time_window_index] = G  # Store the graph

    # Count the frequency of each link across all time windows
    weighted_edgelist_dict = {}
    number_of_windows = len(network_dict.keys())
    for G in network_dict.values():
        for edge in G.edges:
            if edge in weighted_edgelist_dict:
                weighted_edgelist_dict[edge] += 1 / number_of_windows
            else:
                weighted_edgelist_dict[edge] = 1 / number_of_windows

    # Filter links by frequency threshold
    thresholded_edgelist_dict = {}
    for edge, freq in weighted_edgelist_dict.items():
        if freq >= frequency_threshold:
            thresholded_edgelist_dict[edge] = np.round(freq, 2)

    # Create a final graph with edges that meet the frequency threshold
    G_thresh = nx.Graph()
    for edge, weight in thresholded_edgelist_dict.items():
        G_thresh.add_edge(edge[0], edge[1], weight=weight)
    
    # Identify separate groups of connected components
    separate_groups = list(nx.connected_components(G_thresh))

    # Optionally draw the graph
    if draw:
        pos = nx.spring_layout(G_thresh)  # Position nodes using Spring layout
        plt.figure()
        nx.draw(G_thresh, pos, edge_color='black', node_size=1050, with_labels=True, node_color='black', font_color='white')
        nx.draw_networkx_edge_labels(G_thresh, pos, edge_labels=thresholded_edgelist_dict, font_color='red')

    # Filter and return groups larger than 1 asset
    separate_groups_list = [group for group in separate_groups if len(group) > 1]
    return separate_groups_list

def find_dropout_candidates(price_df, time_window_splits_test, drop_out_threshold, separate_groups):
    """
    Identifies potential dropout candidates among assets based on a correlation threshold within specified time windows.
    
    Args:
        price_df (DataFrame): A DataFrame containing price data for various assets.
        time_window_splits_test (list): A list of indices representing the start and end of each time window for testing.
        drop_out_threshold (float): The threshold below which an asset is considered a dropout candidate due to low correlation.
        separate_groups (list): A list of lists, where each inner list contains assets considered part of the same group.

    Returns:
        dict: A dictionary containing the correlation pairs for each time window, keyed by time window index.
    """
    # Initialize dictionary to hold correlation matrices for each time window
    corr_test_df_dict = {}
    for i in range(1, len(time_window_splits_test)):
        # Calculate and store the correlation matrix for each time window
        corr_test_df_dict[i] = price_df.iloc[time_window_splits_test[i-1]:time_window_splits_test[i], :].corr()

    # Extract asset tickers from the correlation matrix
    asset_ticker_list = list(corr_test_df_dict[1].columns)
    # Initialize dictionary to store correlations for all pairs of assets in each time window
    correlation_pairs_test_dict = {}
    for time_window_index in corr_test_df_dict:
        correlation_pairs_test_dict[time_window_index] = {}
        for ticker_1 in asset_ticker_list:
            for ticker_2 in asset_ticker_list:
                if ticker_1 != ticker_2:  # Ensure we're not comparing the same asset
                    # Store the correlation value between each pair of assets
                    correlation_pairs_test_dict[time_window_index][(ticker_1, ticker_2)] = corr_test_df_dict[time_window_index].loc[ticker_1, ticker_2]

    # Prepare to check pairs within the same group for potential dropouts
    pairs_to_check = {}
    for group in separate_groups:
        pairs_to_check[tuple(group)] = []
        for i in group:
            for j in group:
                if j != i:  # Ensure we're not comparing the same asset
                    # Append each unique pair of assets within the group
                    pairs_to_check[tuple(group)].append((i, j))
    
    return correlation_pairs_test_dict,pairs_to_check


def choose_dropout_candidates(price_df,time_window_splits_test,correlation_pairs_test_dict, separate_groups, drop_out_threshold, pairs_to_check):
    """
    This function analyzes correlation data between pairs of assets within specified groups 
    over various time windows to identify potential dropout candidates. A dropout candidate 
    is defined as an asset whose correlation with others in its group falls below a specified 
    threshold, indicating a significant change in its behavior relative to the group.
    
    Args:
        correlation_pairs_test_dict (dict): Dictionary where keys are time window indices and 
            values are dictionaries of asset pairs and their corresponding correlation values.
        separate_groups (list of lists): Each inner list contains assets considered to be in 
            the same group for correlation analysis.
        drop_out_threshold (float): The correlation value below which an asset is considered 
            a dropout candidate.
        pairs_to_check (dict): Precomputed pairs of assets within each group to check for 
            dropouts, keyed by group.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary maps time window 
            indices to potential dropout candidates within each group. The second dictionary 
            contains the chosen dropout candidates after further analysis.
    """
    # Initialize dictionaries for storing potential and chosen dropout candidates
    drop_out_candidates_dict = {}
    drop_out_chosen_dict = {}
    for time_window_index in list(correlation_pairs_test_dict.keys())[:-1]:
        dropout_not_found = False
        drop_out_candidates_dict[time_window_index] = {}
        for group in separate_groups:
            N = len(group)
            drop_out_candidates_dict[time_window_index] [tuple(group)] = {}
            for pair in pairs_to_check[tuple(group)]:
                corr = correlation_pairs_test_dict[time_window_index][pair]
                if corr <= drop_out_threshold:
                    # We add 1/2 because of the double count, will be fixed later
                    if pair[0] in drop_out_candidates_dict[time_window_index][tuple(group)].keys():
                        drop_out_candidates_dict[time_window_index][tuple(group)][pair[0]] += (1/2)/len( pairs_to_check[tuple(group)])
                    else:
                        drop_out_candidates_dict[time_window_index][tuple(group)][pair[0]] = (1/2)/len( pairs_to_check[tuple(group)]) 
                    if pair[1] in drop_out_candidates_dict[time_window_index][tuple(group)].keys():
                        drop_out_candidates_dict[time_window_index][tuple(group)][pair[1]] +=(1/2)/len( pairs_to_check[tuple(group)]) 
                    else:
                        drop_out_candidates_dict[time_window_index][tuple(group)][pair[1]] = (1/2)/len( pairs_to_check[tuple(group)]) 
                #Check for the case if there's no dropout
            #print(drop_out_candidates_dict[time_window_index][tuple(group)])
            if len(drop_out_candidates_dict[time_window_index][tuple(group)]) != 0:
                worst_dropout =  sorted(drop_out_candidates_dict[time_window_index] [tuple(group)].items(),key = lambda x:x[1])[-1][0]
                #print(sorted(drop_out_candidates_dict[time_window_index] [tuple(group)].items(),key = lambda x:x[1]))
                if  drop_out_candidates_dict[time_window_index] [tuple(group)][worst_dropout] >= drop_out_threshold :
                    drop_out_chosen_dict[time_window_index] = {'asset':worst_dropout,'group':group,'time_window_index':time_window_index}
                else:
                    #    #Drop out candiate not found
                    dropout_not_found = True
                #elif :
                     #Drop out candiate not found
                    #dropout_not_found = True
        non_empty_counter = 0
        for group in separate_groups:
            if len(drop_out_candidates_dict[time_window_index] [tuple(group)]) > 0:
                non_empty_counter += 1
                break
        #print(non_empty_counter)
        if  non_empty_counter ==0 or dropout_not_found:
            continue
        start_day = time_window_splits_test[time_window_index-1]-1
        end_day = time_window_splits_test[time_window_index]-1
        start_day_next =  time_window_splits_test[time_window_index]
        end_day_next =  time_window_splits_test[time_window_index+1]
        drop_out_asset = drop_out_chosen_dict[time_window_index]['asset']
        group = drop_out_chosen_dict[time_window_index]['group']
        price_time_window = price_df.iloc[start_day:end_day,:].copy()
        price_time_window_next = price_df.iloc[start_day_next:end_day_next,:].copy()
        
        drop_out_chosen_dict[time_window_index]['trend_change_asset'] = (price_time_window[drop_out_asset].values[-1] - price_time_window[drop_out_asset].values[0])/price_time_window[drop_out_asset].values[0]
        drop_out_chosen_dict[time_window_index]['trend_change_asset_next'] =  (price_time_window_next[drop_out_asset].values[-1] - price_time_window_next[drop_out_asset].values[0])/price_time_window_next[drop_out_asset].values[0]
        if drop_out_chosen_dict[time_window_index]['trend_change_asset'] >= 0:
            drop_out_chosen_dict[time_window_index]['trend_sign_asset'] = 'positive'
        else:
            drop_out_chosen_dict[time_window_index]['trend_sign_asset'] = 'negative'
        drop_out_chosen_dict[time_window_index]['starting_price_asset'] = price_time_window[drop_out_asset].values[0]
        drop_out_chosen_dict[time_window_index]['ending_price_asset'] = price_time_window[drop_out_asset].values[-1]
        drop_out_chosen_dict[time_window_index]['ending_date_asset'] = price_time_window.index[-1]
        drop_out_chosen_dict[time_window_index]['opening_price_asset'] = price_df.iloc[start_day,:][drop_out_asset].copy()
        drop_out_chosen_dict[time_window_index]['opening_date_asset'] = price_time_window.index[-1] +  datetime.timedelta(days = 1)
        group_change_list = []
        group_change_next_list = []
        for asset in group: 
            if asset != drop_out_asset:
                prct_change = (price_time_window[asset].values[-1] - price_time_window[asset].values[0])/price_time_window[asset].values[0]
                prct_change_next = (price_time_window_next[asset].values[-1] - price_time_window_next[asset].values[0])/price_time_window_next[asset].values[0]
                group_change_list.append(prct_change)
                group_change_next_list.append(prct_change_next)
        drop_out_chosen_dict[time_window_index]['trend_change_group'] = np.mean(group_change_list)
        drop_out_chosen_dict[time_window_index]['trend_change_group_next'] = np.mean(group_change_next_list)
        if drop_out_chosen_dict[time_window_index]['trend_change_group'] >= 0:
            drop_out_chosen_dict[time_window_index]['trend_sign_group'] = 'positive'
        else:
            drop_out_chosen_dict[time_window_index]['trend_sign_group'] = 'negative'
    
    return drop_out_chosen_dict

def create_signals(price_df, time_window_splits_test, drop_out_threshold, link_loss_threshold, initial_capital, investment_parameter,separate_groups):
    """
    Creates trading signals based on the analysis of dropout candidates and their correlations within given time windows.
    
    Args:
        price_df (DataFrame): DataFrame containing price data for assets.
        time_window_splits_test (list): Indices marking the start and end of each time window for analysis.
        drop_out_threshold (float): Threshold for considering an asset as a dropout candidate based on correlation.
        link_loss_threshold (float): Not used in this function, but typically for further analysis or filtering.
        initial_capital (float): The initial amount of capital available for investment.
        investment_parameter (float): Parameter to adjust the amount of capital allocated to each trade.
    
    Returns:
        tuple: A tuple containing the dictionary of signals and the updated capital after making all trades.
    """
    # Analyze price data to find dropout candidates and their correlation pairs
    correlation_pairs_test_dict,pairs_to_check = find_dropout_candidates(price_df,time_window_splits_test,drop_out_threshold,separate_groups)
    drop_out_chosen_dict =  choose_dropout_candidates(price_df,correlation_pairs_test_dict, separate_groups, drop_out_threshold, pairs_to_check)
    signals_dictionary = {}
    current_capital = initial_capital

    # Iterate through each chosen dropout to generate investment signals
    for position_index in list(drop_out_chosen_dict.keys()):
        time_window_index = drop_out_chosen_dict[position_index]['time_window_index']
        asset = drop_out_chosen_dict[position_index]['asset']
        group = drop_out_chosen_dict[position_index]['group']
        date = drop_out_chosen_dict[position_index]['opening_date_asset']

        # Only proceed if the opening date is within the data range
        if date < price_df.index[-1]:
            # Calculate the difference in trend change between the group and the asset
            group_asset_change = drop_out_chosen_dict[position_index]['trend_change_group'] - drop_out_chosen_dict[position_index]['trend_change_asset']
            group_asset_change_abs = abs(group_asset_change)
            # Determine investment amount based on the absolute change
            money_to_spend = investment_parameter * current_capital * group_asset_change_abs
            current_capital -= money_to_spend  # Update current capital
            price = drop_out_chosen_dict[position_index]['opening_price_asset']
            volume = int(np.floor(money_to_spend / price))  # Calculate volume to trade

            # Decide position based on the sign of the group asset change
            position = 'long' if group_asset_change_abs > 0 else 'short'
            signals_dictionary[position_index] = {'asset': asset, 'group': group, 'open_volume': volume, 'open_price': price, 'open_date': date, 'position': position}

    # Set a fixed time delta for closing positions
    closing_time_delta = 30

    # Close positions and update capital based on closing prices
    for position_index in list(signals_dictionary.keys()):
        asset = signals_dictionary[position_index]['asset']
        closing_date = signals_dictionary[position_index]['open_date'] + datetime.timedelta(days=closing_time_delta)
        # Ensure the closing date does not exceed the last date in the data
        closing_date = min(closing_date, price_df.index[-1])
        signals_dictionary[position_index]['close_date'] = closing_date
        
        # Fetch the closing price for the asset
        closing_price = price_df.loc[closing_date, asset] if closing_date in price_df.index else price_df.loc[price_df.index[-1], asset]
        signals_dictionary[position_index]['close_price'] = closing_price
        signals_dictionary[position_index]['close_volume'] = signals_dictionary[position_index]['open_volume']
        # Update capital based on the closing transaction
        current_capital += closing_price * signals_dictionary[position_index]['close_volume']

    return signals_dictionary, current_capital

def price_difference_analysis(signals_dictionary, parameters_dictionary, plot):
    """
    Analyze and calculate the cumulative price difference from trading signals, optionally plotting the results.

    This function calculates the cumulative profit or loss for a series of trading positions based on open and close prices,
    and the volume of each trade. It supports both 'long' and 'short' positions. The cumulative profit or loss is calculated
    for each position, and if plotting is enabled, it will display a scatter plot of profits or losses over close dates.

    Parameters:
    - signals_dictionary (dict): A dictionary containing trading signals. Each key represents a position index, and
      its value is another dictionary with details of the trade, including 'position' (long or short), 'open_price',
      'close_price', 'open_volume', and 'close_date'.
    - parameters_dictionary (dict): A dictionary containing parameters for analysis. Currently unused in the function.
    - plot (bool): A boolean indicating whether to plot the cumulative price difference results.

    Returns:
    - float: The final cumulative price difference across all trading positions.
    """
    # Initialize a dictionary to hold the cumulative price difference for each close date.
    price_difference_cumul_dict = {}
    cumulative_price_diff = 0  # Initialize cumulative price difference.

    # Iterate over each position in the signals dictionary.
    for position_index in list(signals_dictionary.keys()):
        # Extract details of the current position from the signals dictionary.
        position = signals_dictionary[position_index]['position']
        close_date = signals_dictionary[position_index]['close_date']
        volume = signals_dictionary[position_index]['open_volume']  # The volume of the position.

        # Calculate the price difference for long positions.
        if position == 'long':
            open_price = signals_dictionary[position_index]['open_price']
            close_price = signals_dictionary[position_index]['close_price']
            cumulative_price_diff += close_price - open_price  # Update cumulative difference.
            price_difference_cumul_dict[close_date] = cumulative_price_diff * volume

        # Calculate the price difference for short positions.
        elif position == 'short':
            open_price = signals_dictionary[position_index]['open_price']
            close_price = signals_dictionary[position_index]['close_price']
            cumulative_price_diff += - (close_price - open_price)  # Update cumulative difference.
            price_difference_cumul_dict[close_date] = cumulative_price_diff * volume
            
    # Plot the results if requested.
    if plot:
        plt.scatter(price_difference_cumul_dict.keys(), price_difference_cumul_dict.values())
        plt.xlabel('Close date')  # X-axis label.
        plt.ylabel('Profit ($)')  # Y-axis label.
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability.

    return cumulative_price_diff  # Return the cumulative price difference.


def create_portfolio(signals_dictionary, start_date, end_date,initial_capital):
    """
    Generates a timeseries portfolio analysis based on trading signals within a specified date range.

    This function processes trading signals to create a portfolio timeseries analysis, tracking the balance, long value,
    short value, and total value of the portfolio over time. It accounts for opening and closing trades based on the
    signals provided and calculates the portfolio's value accordingly.

    Parameters:
    - signals_dictionary (dict): A dictionary where each key is a position index and each value is a dictionary
      containing the details of the trade (asset, position, open_date, close_date, open_price, open_volume).
    - start_date (str): The start date for the analysis in YYYY-MM-DD format.
    - end_date (str): The end date for the analysis in YYYY-MM-DD format.

    Returns:
    - DataFrame: A pandas DataFrame indexed by date, with columns for balance, long value, short value, and total value.
    """
    # Extract unique asset tickers from the signals dictionary.
    ticks = [signals_dictionary[i]['asset'] for i in signals_dictionary]
    ticks = list(set(ticks))  # Remove duplicates.

    # Retrieve price and change data for the assets over the specified date range.
    price_data, change_data = get_data(ticks, start_date, end_date)

    # Create a DataFrame to track portfolio values over the date range.
    date_range = pd.date_range(start_date, end_date)
    portfolio = pd.DataFrame(index=date_range)
    portfolio['balance'] = 0
    portfolio['long value'] = 0
    portfolio['short value'] = 0
    last_price_long, last_price_short = 0, 0 
    cash = initial_capital  # Initialize cash with the initial capital (variable not defined in snippet).
    open_positions = []  # Track open positions.
    
    for t in date_range:
        buy, sell = [], []

        # Determine which positions to buy or sell based on the date.
        for position_index in signals_dictionary:
            if signals_dictionary[position_index]['close_date'] == t:
                sell.append(position_index)
            if signals_dictionary[position_index]['open_date'] == t:
                buy.append(position_index)

        # Update cash and balance for days with no buy/sell activity.
        if len(buy) == 0 and len(sell) == 0:
            portfolio.loc[t, 'balance'] = cash

        # Process buy and sell signals.
        for b in buy:
            position = signals_dictionary[b]['position']
            volume = signals_dictionary[b]['open_volume']
            open_price = signals_dictionary[b]['open_price']

            if position == 'long':
                cash -= open_price * volume
                open_positions.append(b)
            elif position == 'short':
                cash += open_price * volume
                open_positions.append(b)

            portfolio.loc[t, 'balance'] = cash

        for s in sell:
            position = signals_dictionary[s]['position']
            volume = signals_dictionary[s]['open_volume']
            close_price = signals_dictionary[s]['close_price']

            if position == 'long':
                cash += close_price * volume
                open_positions.remove(s)
            elif position == 'short':
                cash -= close_price * volume
                open_positions.remove(s)

            portfolio.loc[t, 'balance'] = cash
        
        # Calculate the value of open long and short positions.
        if open_positions:
            long_value, short_value = 0, 0
            for l in open_positions:
                asset = signals_dictionary[l]['asset']
                volume = signals_dictionary[l]['open_volume']

                if signals_dictionary[l]['position'] == 'long':
                    last_price_long = price_data.loc[t, asset] if t in price_data.index else last_price_long
                    long_value += last_price_long * volume
                    portfolio.loc[t, 'long value'] = long_value

                elif signals_dictionary[l]['position'] == 'short':
                    last_price_short = price_data.loc[t, asset] if t in price_data.index else last_price_short
                    short_value += last_price_short * volume
                    portfolio.loc[t, 'short value'] = short_value

    # Calculate total portfolio value.
    portfolio['total_value'] = portfolio['balance'] + portfolio['long value'] + portfolio['short value']
    
    return portfolio


def risk_free_capital_gain(number_of_days, annual_risk_free_rate, starting_value):
    """
    Calculate the risk-free capital gain over a given number of days.

    This function calculates the gain from an investment at a risk-free rate over a specified
    number of days. The risk-free rate is assumed to be annual, and the calculation adjusts
    it to reflect the period based on the number of days provided. The function returns the
    gain in terms of the increase in value over the starting value, not the final amount.

    Parameters:
    - number_of_days (int): The number of days over which the capital gain is calculated.
    - annual_risk_free_rate (float): The annual risk-free rate of return, expressed as a decimal
      (e.g., 0.05 for 5%).
    - starting_value (float): The initial value of the investment.

    Returns:
    - float: The gain on the starting value after the specified number of days, calculated at the
      risk-free rate. This is the difference between the final value and the starting value.
    """
    # Calculate the final value of the investment after the specified number of days
    # at the given annual risk-free rate, then subtract the starting value to get the gain.
    final_value = (annual_risk_free_rate + 1) ** (number_of_days / 365) * starting_value
    return final_value - starting_value


def strategy_metrics(balance_df, annual_risk_free_rate):
    """
    Calculate and return key performance metrics for a trading strategy compared to a risk-free investment.

    This function computes the total return, total return versus a risk-free investment, and the Sharpe ratio
    for a given trading strategy's performance. It uses a balance DataFrame with a 'total_value' column representing
    the portfolio's value over time, and an annual risk-free rate for comparison.

    Parameters:
    - balance_df (DataFrame): A pandas DataFrame with an index of dates and a 'total_value' column representing
      the daily total value of the portfolio.
    - annual_risk_free_rate (float): The annual risk-free rate of return, expressed as a decimal.

    Returns:
    - dict: A dictionary containing the number of days the strategy was run, the total return of the strategy,
      the total return of the strategy versus a risk-free investment, and the Sharpe ratio.
    """
    # Initial setup: extract starting values and prepare lists for balance tracking.
    initial_capital = balance_df['total_value'].iloc[0]
    date_list = list(balance_df.index)
    balance_list = [initial_capital]
    risk_free_balance_list = [initial_capital]

    # Iterate over each date, updating the current balance and the equivalent risk-free balance.
    for date in date_list:
        current_balance = balance_df['total_value'].loc[date]
        # Calculate next day's risk-free balance based on the last value in the list.
        risk_free_balance_list.append(risk_free_capital_gain(1, annual_risk_free_rate, risk_free_balance_list[-1]))
        balance_list.append(current_balance)

    # Prepare date index for the new DataFrame, starting from the day before the first recorded date.
    zeroth_date = date_list[0] - datetime.timedelta(days=1)
    new_date_list = [zeroth_date] + date_list  # Prepend the starting date.

    # Construct the balance DataFrame with daily returns calculated.
    balance_df = pd.DataFrame({'balance': balance_list}, index=new_date_list)
    balance_df['balance_shift'] = balance_df['balance'].shift(1)  # Shift for return calculation.
    balance_df = balance_df.dropna()  # Remove the first row with NaN.
    balance_df['daily_returns'] = (balance_df['balance'] - balance_df['balance_shift']) / balance_df['balance_shift']

    # Similarly, construct the risk-free balance DataFrame.
    risk_free_balance_df = pd.DataFrame({'risk_free_balance': risk_free_balance_list}, index=new_date_list)
    risk_free_balance_df['risk_free_balance_shift'] = risk_free_balance_df['risk_free_balance'].shift(1)
    risk_free_balance_df = risk_free_balance_df.dropna()
    risk_free_balance_df['risk_free_daily_returns'] = (
        risk_free_balance_df['risk_free_balance'] - risk_free_balance_df['risk_free_balance_shift']
    ) / risk_free_balance_df['risk_free_balance_shift']

    # Calculate total return, total return versus risk-free, and the Sharpe ratio if daily returns' standard deviation is positive.
    total_return = (balance_df['balance'].iloc[-1] - initial_capital) / initial_capital
    total_return_vs_risk_free = (balance_df['balance'].iloc[-1] - risk_free_balance_list[-1]) / risk_free_balance_list[-1]
    if balance_df['daily_returns'].std() > 0:
        sharpe_ratio = (
            balance_df['daily_returns'].mean() - risk_free_balance_df['risk_free_daily_returns'].mean()
        ) / balance_df['daily_returns'].std()
    else:
        sharpe_ratio = np.nan

    # Compile and return the metrics in a dictionary.
    metrics = {
        'number_of_days': len(date_list),
        'total_return': total_return,
        'total_return_vs_risk_free': total_return_vs_risk_free,
        'sharpe_ratio': sharpe_ratio
    }

    return metrics
