# Supply-Demand Forecasting for Ride-Hailing Services

## Project Overview

This project aims to forecast the supply-demand gap for ride-hailing services using regression algorithms. By accurately predicting the gap, ride-hailing companies like Uber can optimize driver utilization, reduce wait times for riders, and minimize surge pricing.

## Background

With less than 10% of the world's citizens owning automobiles, the frequency of commuting via taxis, buses, trains, and planes is high. Uber, a dominant ride-hailing company, processes over 11 million trips, plans over 9 billion routes, and collects over 50TB of data daily. To meet rider needs, Uber must continually innovate in cloud computing, big data technologies, and algorithms to process this massive amount of data and maintain service reliability.

Supply-demand forecasting is critical to maximizing driver utilization and ensuring riders can always get a car when needed. By predicting the volume of drivers and riders in a specific area at a given time, Uber can guide drivers to high-demand areas, resulting in higher earnings for drivers and no surge pricing for riders.

## Problem Definition

A passenger requests a ride by entering the place of origin and destination and clicking "Request Pickup" on the Uber app. A driver answers the request by taking the order. Uber divides a city into non-overlapping square regions and divides one day uniformly into 144 time slots, each 10 minutes long.

- In region \(d_i\) and time slot \(t_j\), the number of passenger requests is denoted as \(r_{ij}\), and the number of driver answers as \(a_{ij}\).
- The demand in region \(d_i\) and time slot \(t_j\) is denoted as \(demand_{ij} = r_{ij}\), and the supply as \(supply_{ij} = a_{ij}\).
- The demand-supply gap is: \(gap_{ij} = r_{ij} - a_{ij}\).

Given the data for any region \(d_i\) and time slot \(t_j\), the goal is to predict \(gap_{ij}\) using a regression algorithm.

## Data Description

### Training Data

The training set contains three consecutive weeks of data for City M in 2016. The goal is to forecast the supply-demand gap for a certain period in the fourth and fifth weeks of City M. The test set contains the data for half an hour before the predicted time slot.

### Tables

#### Order Information Table

| Field Name       | Type   | Description                                   | Example                                  |
|------------------|--------|-----------------------------------------------|------------------------------------------|
| order_id         | string | Order ID                                      | 70fc7c2bd2caf386bb50f8fd5dfef0cf         |
| driver_id        | string | Driver ID (NULL if no driver answered)        | 56018323b921dd2c5444f98fb45509de         |
| passenger_id     | string | User ID                                       | 238de35f44bbe8a67bdea86a5b0f4719         |
| start_region_hash| string | Departure region hash                         | d4ec2125aff74eded207d2d915ef682f         |
| dest_region_hash | string | Destination region hash                       | 929ec6c160e6f52c20a4217c7978f681         |
| price            | double | Price                                         | 37.5                                      |
| time             | string | Timestamp of the order                        | 2016-01-15 00:35:11                      |

#### Region Information Table

| Field Name       | Type   | Description                                   | Example                                  |
|------------------|--------|-----------------------------------------------|------------------------------------------|
| region_hash      | string | Region hash                                   | 90c5a34f06ac86aee0fd70e2adce7d8a         |
| region_id        | string | Region ID                                     | 1                                        |

## Prediction Task

Apply a regression algorithm to predict the supply-demand gap for a given time slot and region. The regressor can be linear or non-linear.

### Output Fields

| Field Name        | Data Type | Example                    |
|-------------------|-----------|----------------------------|
| Region ID         | String    | 1, 2, 3, 4 (same as region ID mapping) |
| Time slot         | String    | 2016-01-23-1 (First time slot on Jan 23, 2016) |
| Prediction value  | Double    | 6.0                        |

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib (for visualization)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mahad002/Supply-Demand-Forecasting.git
   cd Supply-Demand-Forecasting
   ```

2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Prepare your data by placing the Order Information and Region Information tables in the `data/` directory.

2. Run the data preprocessing script:
   ```sh
   python preprocess.py
   ```

3. Train the regression model:
   ```sh
   python train.py
   ```

4. Make predictions:
   ```sh
   python predict.py
   ```

## Results

The model's predictions for the supply-demand gap will be saved in the `results/` directory.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
