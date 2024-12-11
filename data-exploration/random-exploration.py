import pandas as pd
from NTU_ML_Project.visualization_util.plot import visualize_feature_frequency

data = pd.read_csv('../data-for-stage-1/train_data_copy_jason.csv')

def get_home_win_rate(data):
  count = 0
  for index, row in data.iterrows():
    if row['home_team_win']:
      count += 1
  print('total games:', len(data), 'home wins:', count, 'home win rate:', count/(max(len(data),1)))

def get_years(data):
  return sorted(list(set(data['date'].map(lambda x: x[:4]))))

def get_months(data):
  return sorted(list(set(data['date'].map(lambda x: x[5:7]))))

def get_data_by_year(data, year):
  return data[data['date'].map(lambda x: x[:4]) == year]

def get_data_by_month(data, month):
  return data[data['date'].map(lambda x: x[5:7]) == str(month).zfill(2)]

def get_home_win_rate_by_month(data):
  for year in get_years(data):
    year_data = get_data_by_year(data, year)
    for month in get_months(year_data):
      print(year, month)
      data_by_month = get_data_by_month(year_data, month)
      get_home_win_rate(data_by_month)
      visualize_feature_frequency(data_by_month, 'home_team_win')

  
get_home_win_rate_by_month(data)