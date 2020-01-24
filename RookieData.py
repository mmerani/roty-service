# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:43:44 2020

@author: Michael
"""
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from sklearn.externals import joblib
import json

def get_recent_data():
    url = 'https://www.basketball-reference.com/leagues/NBA_2020_rookies.html'
    player_data = []
    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'lxml')
    table = soup.find(attrs={'id':'rookies'})
    roty_table = table.find('tbody')
    rows = roty_table.findAll('tr')
    
    for row in rows:
        if 'class' in row.attrs and row.attrs['class'][0] == 'full_table':
            link = row.find('a')
            td_cells = row.findAll('td')
            player_info = {}
            player_info['link'] = link.get('href') if link else ""
            for index, td in enumerate(td_cells):
                attribute = td['data-stat']
                player_info[attribute] = td.getText()
    
            additional_info = get_additonal_info(player_info['link'])
            player_info['ws'] = additional_info['ws']
            player_info['ws_per_48'] = additional_info['ws_per_48']
            player_info['team'] = additional_info['team'] if additional_info['team'] else "" 
            player_data.append(player_info)
            
    return player_data

def get_additonal_info(player_url):
    #print(player_url)
    if player_url == '':
        return {}
    url = 'https://www.basketball-reference.com' + player_url
    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'lxml')
    advanced_table = soup.find(attrs={'id':'all_advanced'})
    rows = advanced_table.findAll('tr')
    additional_player_info = {}
    
    for child in advanced_table.children:
        if "table_outer_container" in child:
            other_soup = BeautifulSoup(child)
            rows = other_soup.findAll("tr")

    for row in rows:
        if 'id' in row.attrs and row.attrs['id'] == 'advanced.2020':
            
            additional_player_info['ws'] = float(row.find('td', attrs={'data-stat': 'ws'}).text)
            additional_player_info['ws_per_48'] = float(row.find('td', attrs={'data-stat': 'ws_per_48'}).text)
            break
        
    player_info = soup.findAll("p")
    for p in player_info:
        strong = p.find('strong')
        if strong and strong.getText() == 'Team':
            additional_player_info['team'] = p.find('a').text
            #print(p.find('a').text)
            break
        
    if 'team' not in additional_player_info:
        additional_player_info['team'] = ''
        
    #print(player_win_shares)
    return additional_player_info

    

def create_rookie_df():
    
    rookies = get_recent_data()

    rookies_df = pd.DataFrame(rookies)
    #features = ['ast_per_g','blk_per_g', 'fg3_pct', 'fg_pct', 'mp_per_g', 'pts_per_g','stl_per_g',
    #           'ws','ws_per_48']
    rookies_df = rookies_df.drop(['age','debut', 'years'],axis = 1)
    rookies_df['mp'] = rookies_df['mp'].fillna("0")
    rookies_df = rookies_df[rookies_df['mp'].astype('int') >= 500]
    
    rookies_df['ast_per_g'] = rookies_df.ast_per_g.astype(float)
    rookies_df['fg3_pct'] = rookies_df.fg3_pct.astype(float)
    rookies_df['fg_pct'] = rookies_df.fg_pct.astype(float)
    rookies_df['mp_per_g'] = rookies_df.mp_per_g.astype(float)
    rookies_df['pts_per_g'] = rookies_df.pts_per_g.astype(float)
    rookies_df['blk'] = rookies_df.blk.astype(int)
    rookies_df['stl'] = rookies_df.stl.astype(int)
    rookies_df['g'] = rookies_df.g.astype(int)
    
    return rookies_df

def predict_roty():
    lr = joblib.load('classifiers/Linear_Regression.joblib')
    gd = joblib.load('classifiers/Gradient_Descent.joblib')
    ridge = joblib.load('classifiers/Ridge_Regression.joblib')
    lasso = joblib.load('classifiers/Lasso_Regression.joblib')
    elastic = joblib.load('classifiers/Elastic_Net.joblib')
    
    models = [lr,gd,ridge,lasso,elastic]
    
    current_rookies_df = create_rookie_df()
    features = ['ast_per_g','blk_per_g', 'fg3_pct', 'fg_pct', 'mp_per_g', 'pts_per_g','stl_per_g',
           'ws','ws_per_48']
    current_rookies_df['blk_per_g'] = current_rookies_df['blk'] / current_rookies_df['g']
    current_rookies_df['stl_per_g'] = current_rookies_df['stl'] / current_rookies_df['g']
    
    rookies = current_rookies_df[features]
    model_names = ['Linear Regression', 'Gradient Descent', 'Ridge Regression', 'Lasso Regression','Elastic Net']
    count = 0
    model_predictions = {}
    
    for model in models:
        rookies_predict_y = model.predict(rookies)
        sorted_indices = np.argsort(rookies_predict_y)[::-1]
        
        rookies_list = []
        for i in range(0,5):
            rookies_list.append({'Player': current_rookies_df.iloc[sorted_indices[i]].player,
                                 'Team': current_rookies_df.iloc[sorted_indices[i]].team})
        
        model_predictions[model_names[count]] = rookies_list
        count += 1
    
    with open('predictions.json', 'w') as f:
        json.dump(model_predictions, f)
        

predict_roty()
