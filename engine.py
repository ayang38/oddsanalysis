import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

DATAFOLDER = 'DATAFOLDER'
OUTPUT_FOLDER = 'ANALYSIS_OUTPUT'
PROCESSED_DATA_FILE = 'processed_odds_data.csv'
ARBITRAGES_FILE = 'arbitrages.csv'
BANKROLL_GROWTH_FILE = 'bankroll_growth.csv'
REALISTICNESS_PERCENT = 10

def load_all_snapshots():
    data_rows = []
    for filename in os.listdir(DATAFOLDER):
        if filename.endswith('.json'):
            path = os.path.join(DATAFOLDER, filename)
            timestamp = datetime.fromtimestamp(os.path.getmtime(path))
            with open(path, 'r') as f:
                snapshot = json.load(f)
            for event in snapshot:
                event_id = event.get('id')
                sport_key = event.get('sport_key')
                sport_title = event.get('sport_title')
                commence_time = event.get('commence_time')
                home_team = event.get('home_team')
                away_team = event.get('away_team')
                for bookmaker in event.get('bookmakers', []):
                    bookmaker_key = bookmaker.get('key')
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')
                        for outcome in market.get('outcomes', []):
                            outcome_name = outcome.get('name')
                            price = outcome.get('price')
                            point = outcome.get('point', np.nan)
                            data_rows.append({
                                'timestamp': timestamp,
                                'event_id': event_id,
                                'sport_key': sport_key,
                                'sport_title': sport_title,
                                'commence_time': commence_time,
                                'home_team': home_team,
                                'away_team': away_team,
                                'bookmaker_key': bookmaker_key,
                                'market_key': market_key,
                                'outcome_name': outcome_name,
                                'price': price,
                                'point': point
                            })


    df = pd.DataFrame(data_rows)
    if df.empty:
        print("No data loaded. Check if JSON files exist and are valid.")
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['commence_time'] = pd.to_datetime(df['commence_time'])
    df['commence_time'] = df['commence_time'].dt.tz_convert(None)
    df.sort_values(by=['event_id', 'timestamp', 'bookmaker_key', 'market_key'], inplace=True)
    return df

def save_processed_data(df):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, PROCESSED_DATA_FILE)
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

def detect_arbitrages(df):
    arbs = []
    h2h_df = df[df['market_key'] == 'h2h']
    grouped = h2h_df.groupby(['event_id', 'timestamp'])
    for (event_id, timestamp), group in grouped:
        commence_time = group['commence_time'].iloc[0]  
        status = 'pregame' if timestamp < commence_time else 'in-game'
        outcomes = group.groupby('outcome_name')['price'].max()
        if len(outcomes) == 2:  
            outcome1, outcome2 = outcomes.index
            max_price1 = outcomes[outcome1]
            max_price2 = outcomes[outcome2]
            arb_value = (1 / max_price1) + (1 / max_price2)
            if arb_value < 1:
                profit = (1 - arb_value) * 100
                if profit <= REALISTICNESS_PERCENT:
                    team1 = outcome1
                    team2 = outcome2
                    arbs.append({
                        'event_id': event_id,
                        'timestamp': timestamp,
                        'team1': team1,
                        'team2': team2,
                        'max_price1': max_price1,
                        'max_price2': max_price2,
                        'arb_score': arb_value,
                        'potential_profit_percent': profit,
                        'status': status,
                        'commence_time': commence_time
                    })
    arb_df = pd.DataFrame(arbs)
    if not arb_df.empty:
        arb_df.sort_values(by='potential_profit_percent', ascending=False, inplace=True)
        arb_path = os.path.join(OUTPUT_FOLDER, ARBITRAGES_FILE)
        arb_df.to_csv(arb_path, index=False)
        print(f"Arbitrages detected and saved to {arb_path}")
    else:
        print("No arbitrages detected.")
    return arb_df

def simulate_arbitrage(stake, arb_row):
    odds1 = arb_row['max_price1']
    odds2 = arb_row['max_price2']
    arb_value = arb_row['arb_score']
    if arb_value >= 1:
        return stake, 0  # No arb
    denom = arb_value
    stake1 = stake * (1 / odds1) / denom
    stake2 = stake * (1 / odds2) / denom
    # profit = whichever wins
    payout1 = stake1*odds1
    payout2 = stake2*odds2
    payout = min(payout1, payout2)
      #will just do lowest for now to simulate  mininmuim gains
    profit = payout - stake
    return profit

def plot_bankroll_growth(bankrolls):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(bankrolls)), bankrolls, marker='o', linestyle='-', color='b')
    plt.title('Bankroll Growth Over Arbitrages')
    plt.xlabel('Number of Arbitrages')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_FOLDER, 'bankroll_growth.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Bankroll growth plot saved to {plot_path}")

def save_bankroll_growth_csv(growth_data):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    growth_df = pd.DataFrame(growth_data)
    csv_path = os.path.join(OUTPUT_FOLDER, BANKROLL_GROWTH_FILE)
    growth_df.to_csv(csv_path, index=False)
    print(f"Bankroll growth saved to {csv_path}")

def plot_odds_changes(df, event_id):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    event_df = df[df['event_id'] == event_id]
    if event_df.empty:
        print(f"No data for event {event_id}.")
        return
    pivot = event_df.pivot_table(index='timestamp', columns=['bookmaker_key', 'outcome_name'], values='price', aggfunc='first')
    plt.figure(figsize=(12, 6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=str(col))
    plt.title(f'Odds Changes Over Time for Event {event_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Decimal Odds')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    commence_time = event_df['commence_time'].iloc[0]
    plt.axvline(x=commence_time, color='red', linestyle='--', label='Game Start')
    plot_path = os.path.join(OUTPUT_FOLDER, f'odds_changes_{event_id}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Odds changes plot for event {event_id} saved to {plot_path}")

def odds_change_analysis(df):
    # events with both pre-game and in-game data
    grouped = df.groupby('event_id')
    qualifying_events = []
    for event_id, group in grouped:
        commence_time = group['commence_time'].iloc[0]
        has_pre = (group['timestamp'] < commence_time).any()
        has_in = (group['timestamp'] >= commence_time).any()
        if has_pre and has_in:
            qualifying_events.append(event_id)
    if not qualifying_events:
        print("No events with both pre-game and in-game data.")
        return
    print(f"Found {len(qualifying_events)} qualifying events.")
    for event_id in qualifying_events:
        plot_odds_changes(df, event_id)

def print_analysis_summary(df, arb_df):
    total_events = len(df.groupby(['event_id', 'timestamp']))
    total_arbitrages = len(arb_df)
    if not df.empty:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        days_span = (max_date - min_date).days + 1  # Inclusive
    else:
        days_span = 0


    average_arbs_per_day = total_arbitrages / days_span if days_span > 0 else 0
    average_profit_percent = arb_df['potential_profit_percent'].mean() if not arb_df.empty else 0
    print(f"Total events analyzed: {total_events}")
    print(f"Total arbitrages found: {total_arbitrages}")
    print(f"Data collection span: {days_span} days")
    print(f"Average arbitrages per day: {average_arbs_per_day:.2f}")
    print(f"Average profit per arbitrage: {average_profit_percent:.2f}%")
    return average_arbs_per_day, average_profit_percent

def live_simulator(initial_bankroll):
    bankroll = initial_bankroll
    max_stake_per_bet = initial_bankroll  
    print(f"Starting live simulator with initial bankroll: {initial_bankroll}, max stake per bet: {max_stake_per_bet}")
    processed_files = set(os.listdir(DATAFOLDER))
    seen_arbs = set()  #fixes same specific bet displaying 100x, will make data clearer
    try:
        while True:
            current_files = set(os.listdir(DATAFOLDER))
            new_files = current_files - processed_files
            if new_files:
                print(f"Detected {len(new_files)} new snapshots.")
                new_data_rows = []
                for filename in new_files:
                    if filename.endswith('.json'):
                        path = os.path.join(DATAFOLDER, filename)
                        timestamp = datetime.fromtimestamp(os.path.getmtime(path))
                        with open(path, 'r') as f:
                            snapshot = json.load(f)
                        for event in snapshot:
                            event_id = event.get('id')
                            sport_key = event.get('sport_key')
                            sport_title = event.get('sport_title')
                            commence_time = event.get('commence_time')
                            home_team = event.get('home_team')
                            away_team = event.get('away_team')
                            for bookmaker in event.get('bookmakers', []):
                                bookmaker_key = bookmaker.get('key')
                                for market in bookmaker.get('markets', []):
                                    market_key = market.get('key')
                                    for outcome in market.get('outcomes', []):
                                        outcome_name = outcome.get('name')
                                        price = outcome.get('price')
                                        point = outcome.get('point', np.nan)
                                        new_data_rows.append({
                                            'timestamp': timestamp,
                                            'event_id': event_id,
                                            'sport_key': sport_key,
                                            'sport_title': sport_title,
                                            'commence_time': commence_time,
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'bookmaker_key': bookmaker_key,
                                            'market_key': market_key,
                                            'outcome_name': outcome_name,
                                            'price': price,
                                            'point': point
                                        })

                if new_data_rows:
                    new_df = pd.DataFrame(new_data_rows)
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
                    new_df['commence_time'] = pd.to_datetime(new_df['commence_time'])
                    new_df['commence_time'] = new_df['commence_time'].dt.tz_convert(None)
                    # detects new arbs in new data
                    new_arb_df = detect_arbitrages(new_df)
                    for _, arb_row in new_arb_df.iterrows():
                        arb_key = (arb_row['event_id'], arb_row['timestamp'])
                        if arb_key not in seen_arbs:
                            seen_arbs.add(arb_key)
                            print(f"New arbitrage detected: {arb_row['team1']} ({arb_row['max_price1']}) vs {arb_row['team2']} ({arb_row['max_price2']}), profit %: {arb_row['potential_profit_percent']:.2f}")
                            stake = min(bankroll, max_stake_per_bet)
                            profit = simulate_arbitrage(stake, arb_row)
                            print(f"Simulated bet with stake {stake:.2f}: Profit {profit:.2f}")
                            bankroll += profit
                            print(f"New bankroll: {bankroll:.2f}")
                processed_files = current_files
            time.sleep(60)  #CHECK EVERY ???
    except KeyboardInterrupt:
        print(f"Exiting live simulator. Final bankroll: {bankroll:.2f}")

def historical_simulation(initial_bankroll):
    df = load_all_snapshots()
    arb_df = detect_arbitrages(df)
    average_arbs_per_day, average_profit_percent = print_analysis_summary(df, arb_df)
    bankroll = initial_bankroll
    max_stake_per_bet = initial_bankroll  # each bank roll bet doesn't go over ? -- keeps bets realistic
    growth_data = [{'Arbitrage_Number': 0, 'Event_Name': 'Initial', 'Profit_Percent': 0.0, 'Bankroll': initial_bankroll}]


    print(f"Starting historical simulation with initial bankroll: {initial_bankroll}, max stake per bet: {max_stake_per_bet}")
    for i, arb_row in enumerate(arb_df.iterrows(), start=1):
        _, arb_row = arb_row
        stake = min(bankroll, max_stake_per_bet)
        profit = simulate_arbitrage(stake, arb_row)
        bankroll += profit
        event_name = f"{arb_row['team1']} vs {arb_row['team2']}"
        growth_data.append({
            'Arbitrage_Number': i,
            'Event_Name': event_name,
            'Profit_Percent': arb_row['potential_profit_percent'],
            'Bankroll': bankroll
        })
    print(f"Final bankroll after all arbitrages: {bankroll:.2f}")
    num_arbs = len(arb_df)
    total_profit = bankroll - initial_bankroll
    average_profit_dollars = total_profit / num_arbs if num_arbs > 0 else 0
    print(f"Average profit per arbitrage (based on bets): ${average_profit_dollars:.2f}")
    save_bankroll_growth_csv(growth_data)
    plot_bankroll_growth([entry['Bankroll'] for entry in growth_data])
    simulate_future = input("Do you want to simulate returns for the next X days? (yes/no): ").strip().lower()

    if simulate_future == 'yes':
        try:
            future_days = int(input("Enter the number of days: ").strip())
            predicted_arbs = average_arbs_per_day * future_days
            predicted_profit = predicted_arbs * average_profit_dollars
            predicted_bankroll = bankroll + predicted_profit
            print(f"Predicted arbitrages over {future_days} days: {predicted_arbs:.2f}")
            print(f"Predicted total profit: ${predicted_profit:.2f}")
            print(f"Predicted final bankroll: ${predicted_bankroll:.2f}")
        except ValueError:
            print("Invalid input for days.")

if __name__ == "__main__":

    print("Choose an option:")
    print("1: Live simulator (monitors DATAFOLDER for new files and simulates bets on new arbs)")
    print("2: Historical simulation (simulates profits on all existing data)")
    print("3: Odds change analysis (plot odds changes for qualifying events)")
    choice = input("Enter your choice: ").strip()

    if choice in ['1', '2', '3']:
        if choice in ['1', '2']:
            bankroll_input = input("Enter the initial bankroll amount: ").strip()
            try:
                initial_bankroll = float(bankroll_input)
            except ValueError:
                print("Invalid input. Using default bankroll of 1000.0")
                initial_bankroll = 1000.0
        if choice == '1':
            live_simulator(initial_bankroll)
        elif choice == '2':
            historical_simulation(initial_bankroll)
        elif choice == '3':
            df = load_all_snapshots()
            odds_change_analysis(df)
    else:
        print("Invalid choice. Exiting.")
