import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pydeck as pdk

rent_data = {
    'chennai': {'1bhk': {'advance': 40000, 'rent': 8000},
                '2bhk': {'advance': 2000, 'rent': 10000},
                '3bhk': {'advance': 4000, 'rent': 15000}},
    'trichy': {'1bhk': {'advance': 10000, 'rent': 3000},
               '2bhk': {'advance': 1000, 'rent': 7000},
               '3bhk': {'advance': 3000, 'rent': 9000}},
    'mumbai': {'1bhk': {'advance': 40000, 'rent': 21000},
               '2bhk': {'advance': 2000, 'rent': 30000},
               '3bhk': {'advance': 5000, 'rent': 45000}},
    'bangalore': {'1bhk': {'advance': 50000, 'rent': 10000},
                  '2bhk': {'advance': 7000, 'rent': 15000},
                  '3bhk': {'advance': 9000, 'rent': 20000}},
    'kochi': {'1bhk': {'advance': 20000, 'rent': 5000},
              '2bhk': {'advance': 6000, 'rent': 8000},
              '3bhk': {'advance': 3000, 'rent': 12000}},
    'mangalore': {'1bhk': {'advance': 30000, 'rent': 7000},
                  '2bhk': {'advance': 1000, 'rent': 10000},
                  '3bhk': {'advance': 2000, 'rent': 15000}},
    'coimbatore': {'1bhk': {'advance': 20000, 'rent': 6000},
                   '2bhk': {'advance': 4000, 'rent': 9000},
                   '3bhk': {'advance': 9000, 'rent': 12000}},
    'pune': {'1bhk': {'advance': 30000, 'rent': 9000},
             '2bhk': {'advance': 4000, 'rent': 12000},
             '3bhk': {'advance': 2000, 'rent': 18000}},
    'delhi': {'1bhk': {'advance': 50000, 'rent': 15000},
              '2bhk': {'advance': 3000, 'rent': 20000},
              '3bhk': {'advance': 1000, 'rent': 25000}}
}

tenant_data = {'bachelor': {'1bhk': {'rent': 8000},
                            '2bhk': {'rent': 10000},
                            '3bhk': {'rent': 15000}},
               'family': {'1bhk': {'rent': 6000},
                          '2bhk': {'rent': 8000},
                          '3bhk': {'rent': 13000}}
               }

def train_model():
    df = pd.DataFrame(rent_data).T
    df = df.stack().apply(pd.Series).stack().unstack(level=-1).reset_index()
    df.columns = ['place', 'config', 'advance', 'rent']
    encoder = LabelEncoder()
    df['config_encoded'] = encoder.fit_transform(df['config'])
    X = df[['config_encoded']]
    y = df['rent']
    model = LinearRegression()
    model.fit(X, y)

    return model

model = train_model()

def main():
    st.title('House Rent Price Prediction')
    st.subheader('Input')
    place = st.selectbox('Place', ('chennai', 'trichy', 'mumbai', 'bangalore', 'kochi', 'mangalore', 'coimbatore', 'pune', 'delhi'))
    tenant_type = st.selectbox('Tenant Type', ('bachelor', 'family'))
    num_rooms = st.selectbox('Number of Rooms', (1, 2, 3))
    if tenant_type == 'bachelor':
        advance = rent_data[place][f'{num_rooms}bhk']['advance']
        rent = rent_data[place][f'{num_rooms}bhk']['rent']
    else:
        advance = tenant_data[tenant_type][f'{num_rooms}bhk'].get('advance', 0)
        rent = tenant_data[tenant_type][f'{num_rooms}bhk']['rent']
    st.subheader('Rent Prediction')
    st.write(f'Advance Amount: {advance}')
    st.write(f'Rent Per Month: {rent}')
    st.subheader('Location')
    location = get_location(place)
    map = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state={
            'latitude': location[0],
            'longitude': location[1],
            'zoom': 12,
            'pitch': 50,
        },
        layers=[pdk.Layer('ScatterplotLayer', data=[{'position': location, 'text': place}], get_position='position', get_text='text', get_color=[255, 0, 0], get_radius=1000)],
    )
    st.pydeck_chart(map)
def get_location(place):
    location_data = {
        'chennai': (13.0827, 80.2707),
        'trichy': (10.7905, 78.7047),
        'mumbai': (19.0760, 72.8777),
        'bangalore': (12.9716, 77.5946),
        'kochi': (9.9312, 76.2673),
        'mangalore': (12.9141, 74.8559),
        'coimbatore': (11.0168, 76.9558),
        'pune': (18.5204, 73.8567),
        'delhi': (28.7041, 77.1025)
    }
    return location_data[place]
if __name__ == '__main__':
    main()
