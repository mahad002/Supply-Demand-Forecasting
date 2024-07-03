import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime, timedelta

def load_data(cluster_map_path, order_data_dir):
    cluster_columns = ['region_hash', 'region_id']
    order_columns = ['order_id', 'driver_id', 'passenger_id', 'start_region_hash', 'dest_region_hash', 'Price', 'Time']

    cluster_map_df = pd.read_csv(cluster_map_path, delimiter='\t', header=None, names=cluster_columns)

    order_data_df = pd.DataFrame()

    start_date = datetime(2016, 1, 1)
    end_date = datetime(2016, 1, 21)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    for date in date_range:
        filename = f"order_data_{date.strftime('%Y-%m-%d')}"
        file_path = os.path.join(order_data_dir, filename)
        if os.path.exists(file_path):  
            current_data = pd.read_csv(file_path, delimiter='\t', header=None, names=order_columns)
            order_data_df = pd.concat([order_data_df, current_data], ignore_index=True)
        else:
            print(f"File not found: {filename}")

    return cluster_map_df, order_data_df

def divide_data_into_groups(cluster_map_df, order_data_df):
    try:
        print("Dividing data into groups...")

        order_data_df = order_data_df.merge(cluster_map_df, left_on='start_region_hash', right_on='region_hash', how='left')
        order_data_df = order_data_df.rename(columns={'region_id': 'start_region_id'})

        order_data_df = order_data_df.merge(cluster_map_df, left_on='dest_region_hash', right_on='region_hash', how='left')
        order_data_df = order_data_df.rename(columns={'region_id': 'dest_region_id'})

        order_data_df['Time'] = pd.to_datetime(order_data_df['Time'])

        order_data_df['hour'] = order_data_df['Time'].dt.hour
        order_data_df['minute'] = order_data_df['Time'].dt.minute
        order_data_df['day_of_week'] = order_data_df['Time'].dt.dayofweek

        order_data_df['time_slot'] = (order_data_df['hour'] * 60 + order_data_df['minute']) // 10

        demand_supply_df = order_data_df.groupby(['start_region_id', 'time_slot', 'day_of_week']).agg({'driver_id': lambda x: x.isnull().sum(), 'passenger_id': 'count'}).reset_index()
        demand_supply_df.columns = ['region_id', 'time_slot', 'day_of_week', 'supply', 'demand']

        demand_supply_df['demand_supply_gap'] = demand_supply_df['demand'] - demand_supply_df['supply']

        print("Data divided into groups successfully.")
        return demand_supply_df
    except Exception as e:
        print(f"Error dividing data into groups: {e}")
        return None

def split_data(demand_supply_df):
    try:
        print("Splitting data into training and testing sets...")

        X = demand_supply_df[['region_id', 'time_slot']]
        y = demand_supply_df['demand_supply_gap']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Data split successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    try:
        print("Training the regression model...")

        model = LinearRegression()

        model.fit(X_train, y_train)

        print("Model trained successfully.")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    try:
        print("Evaluating the regression model...")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        return y_pred

    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def predict_test_data(model, X_test, output_csv):
    try:
        print("Predicting test data...")

        y_pred = model.predict(X_test)

        X_test['Prediction'] = y_pred

        X_test['region_id'] = X_test['region_id'].astype(str)
        X_test['time_slot'] = X_test['time_slot'].astype(str)

        X_test['Time slot'] = X_test['time_slot'] + "-" + X_test['region_id']

        X_test[['region_id', 'time_slot', 'Prediction']].to_csv(output_csv, index=False)

        print(f"Predictions saved to {output_csv}")

    except Exception as e:
        print(f"Error predicting test data: {e}")

def main():
    try:
        # Define paths
        cluster_map_path = 'training_data/cluster_map/cluster_map'  
        order_data_dir = 'training_data/order_data/'         
        output_csv = 'predictions.csv'  

        cluster_map_df, order_data_df = load_data(cluster_map_path, order_data_dir)
        print("Cluster map data:")
        print(cluster_map_df)
        print("Order data:")
        print(order_data_df)

        demand_supply_df = divide_data_into_groups(cluster_map_df, order_data_df)
        if demand_supply_df is not None:
            print("Demand and Supply data:")
            print(demand_supply_df)

            X_train, X_test, y_train, y_test = split_data(demand_supply_df)
            if X_train is not None and X_test is not None and y_train is not None and y_test is not None:

                model = train_model(X_train, y_train)
                if model is not None:

                    y_pred = evaluate_model(model, X_test, y_test)

                    if y_pred is not None:
                        predict_test_data(model, X_test, output_csv)

                else:
                    print("Error: Model not trained.")

            else:
                print("Error: Data split failed.")

        else:
            print("Error: Data division failed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
