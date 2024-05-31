"""
Name: Edgar Campos
CS230: Section 1
Data: Boston Blue Bikes
URL:

"""
# Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#import seaborn as sns
import streamlit as st
from math import radians, sin, cos, sqrt, atan2
import pydeck as pdk
#Simport mapbox as mb
from matplotlib.lines import Line2D 

trips_file = '201501-hubway-tripdata.csv'
bluebikes_image = 'https://facilities.northeastern.edu/wp-content/uploads/2021/12/Bluebikes-Pic-4-1860x970.jpg'
bluebikes_image2 = 'https://img.mlbstatic.com/mlb-images/image/private/t_16x9/t_w1536/mlb/axbuxlnz5y2irpqyjsvn.jpg'
research_center = 'edgarscientific.png'
advertisement = 'edgarbikead.png'

# Function to load in the dataset
def load_data(file_name):
    df = pd.read_csv(file_name)
    return df

# Function used to clean up our user dataset
def clean_trips(trip_data):

    # Remove \N values in data
    to_replace = {'\\N': np.nan}
    trips = trip_data.replace(to_replace)
    trips.dropna(inplace=True)

    # Create Age column (data is from 2015) by subtracting from birth year
    trips = trips.astype({'birth year': 'int64'})
    trips['age'] = 2015 - trips['birth year']

    # Calculate trip distance
    trips['distance (mi)'] = trips.apply(lambda row: calc_distance(row['start station latitude'],
                                                                     row['start station longitude'],
                                                                     row['end station latitude'],
                                                                     row['end station longitude']), axis=1)
    
    trips['distance (km)'] = trips.apply(lambda row: calc_distance(row['start station latitude'],
                                                                     row['start station longitude'],
                                                                     row['end station latitude'],
                                                                     row['end station longitude'],units='km'), axis=1)
                                                                     
    # Convert starttime to date and time format
    trips[['date','time']] = trips['starttime'].str.split(pat=' ',n=1,expand=True)
    trips['date'] = pd.to_datetime(trips['date'])
    trips['dayoftheweek'] = trips['date'].dt.day_name()
    trips[['hour','minute','second']] = trips['time'].str.split(pat=':',n=4, expand=True)


    # Convert trip duration from seconds to minutes
    trips['duration (min)'] = trips['tripduration'] / 60

    # Remove unused columns
    to_drop = ['birth year', 'start station id', 'end station id', 'bikeid', 'gender', 
               'tripduration', 'minute', 'second', 'starttime', 'stoptime']

    trips = trips.drop(columns=to_drop).reset_index(drop=True)
    
    return trips

# Function to clean data on station locations
def clean_stations(trip_data):

    # Create a DataFrame of all unique start and end stations
    start = trip_data[['start station name',
                       'start station latitude',
                       'start station longitude']].drop_duplicates()
    
    end = trip_data[['end station name',
                     'end station latitude',
                     'end station longitude']].drop_duplicates()
    
    # Create a dict with all the column names and what we want in the final DataFrame
    column_Rename = {
        'start station name': 'station_name',
        'start station latitude': 'lat',
        'start station longitude': 'lon',
        'end station name': 'station_name',
        'end station latitude': 'lat',
        'end station longitude': 'lon'
    }

    # Rename columns
    start = start.rename(columns=column_Rename)
    end = end.rename(columns=column_Rename)

    # Combine two DataFrames and remove duplicates
    stations = pd.concat([start,end]).drop_duplicates()
    stations = stations.sort_values(by='station_name').reset_index(drop=True)

    return stations

def calc_distance(lat1,lon1,lat2,lon2,units='mi'):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in miles and kilometers (change if needed)
    if units == 'mi':
        R = 3958.8
    elif units == 'km':
        R = 6371.0
    else:
        return ValueError('Invalid unit type')
    
    # Calculate the distance
    distance = R * c

    return distance

# Function used to calculate trip distance between two station's coordinates
def trip_distance(starting,ending,df,units='mi'):

    lat1 = df.loc[(df['station_name']==starting),'lat'].values[0]
    lon1 = df.loc[(df['station_name']==starting),'lon'].values[0]
    lat2 = df.loc[(df['station_name']==ending),'lat'].values[0]
    lon2 = df.loc[(df['station_name']==ending),'lon'].values[0]

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in miles and kilometers (change if needed)
    if units == 'mi':
        R = 3958.8
    elif units == 'km':
        R = 6371.0
    else:
        return ValueError('Invalid unit type')
    
    # Calculate the distance
    distance = R * c

    return distance

# Function used to create a regression model to find the average time a person takes to travel a distance
def simple_regression(x, y):

    # Number of observations/points
    n = np.size(x)
    
    # Mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    # Calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    # Calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    return (b_0, b_1) 

def station_map(df):
    # Create a subheader for the station map
    st.subheader(':blue[Locations of Blue Bike Stations]')

    # Sets the default view
    view_state = pdk.ViewState(
        latitude = df['lat'].mean(),
        longitude = df['lon'].mean(),
        zoom = 11.5
    )

    # Sets the station layer on top of the map
    station_layer = pdk.Layer(
        'ScatterplotLayer',
        data=df,
        get_position='[lon, lat]',
        get_radius='scaled_radius',
        radius_scale = 2,
        radius_min_pixels= 5,
        radius_max_pixels = 15,
        get_color=[0,0,255],
        pickable=True
    )

    # Create the tooltip to each station
    tool_tip = {
        "html": "Station Name:<br/> <b>{station_name}</b> ",
        "style": { "backgroundColor": 'lightblue',"color": "white"}
    }

    # Create the map setings
    map = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[station_layer],
        tooltip=tool_tip
    )

    # Create the map in streamlit
    st.pydeck_chart(map)


def trip_map(starting, ending, df):

    df_trip = df.loc[(df['station_name']==starting)|(df['station_name']==ending)]

    if len(df_trip) == 2:
        start_coords = df_trip[df_trip['station_name'] == starting][['lon', 'lat']].values[0]
        end_coords = df_trip[df_trip['station_name'] == ending][['lon', 'lat']].values[0]
        line_data = pd.DataFrame({
            'start_lon': [start_coords[0]],
            'start_lat': [start_coords[1]],
            'end_lon': [end_coords[0]],
            'end_lat': [end_coords[1]]
        })

        trip_layer = pdk.Layer(
                'LineLayer',
                data=line_data,
                get_source_position='[start_lon, start_lat]',
                get_target_position='[end_lon, end_lat]',
                get_color=[0,0,255],
                get_width=5,
                pickable=False
            )
    
        trip_state = pdk.ViewState(
            latitude = df_trip['lat'].mean(),
            longitude = df_trip['lon'].mean(),
            zoom = 12.5
        )

        station_layer = pdk.Layer(
            'ScatterplotLayer',
            data=df_trip,
            get_position='[lon, lat]',
            get_radius='scaled_radius',
            radius_scale = 2,
            radius_min_pixels= 5,
            radius_max_pixels = 15,
            get_color=[0,0,255],
            pickable=True
        )

        tool_tip = {
            "html": "Station Name:<br/> <b>{station_name}</b> ",
            "style": { "backgroundColor": 'lightblue',"color": "white"}
        }

        trip_map = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=trip_state,
            layers=[station_layer,trip_layer],
            tooltip=tool_tip
        )
        
        st.pydeck_chart(trip_map)
    else:
        station_map(df)


def main():
    # Configure website
    st.set_page_config(page_title='CS230 Final',
                        page_icon='🚲',
                        layout='centered',
                        initial_sidebar_state='collapsed')
    
    # Load in data
    df_original = load_data(trips_file)
    df_stations = clean_stations(df_original)
    df_trips = clean_trips(df_original)

    # Set up a sidebar for feedback along with an advertisement
    st.sidebar.write('Give us a review!')
    st.sidebar.slider('Leave a rating!', min_value = 0, max_value = 10)
    st.sidebar.radio('How did you hear about us?',['Social Media','Advertisement','Friends/Family'])
    st.sidebar.text_input('Additional comments?')
    sidebutton = st.sidebar.button('Submit')

    if sidebutton:
        st.sidebar.write('Thank you for your feedback!')

    st.sidebar.image(advertisement,caption='Ad powered by Edgar Campos')

    st.image(research_center)

    tab1, tab2, tab3 = st.tabs(['Home',
                                'Trip Planner',
                                'Analysis'])
    
    with tab1:
        st.warning('The data presented is relevant to Boston Blue Bikes as of January of 2015')
        st.header(':blue[Boston Blue Bikes]')
        st.image(bluebikes_image)

        bluebike_description = ('Boston Blue Bikes is a bike-sharing program in Boston, Massachusetts. It allows users to rent bicycles for short trips around the city. ' 
        'The program provides a convenient and environmentally friendly transportation option for residents and visitors. ' 
        'Users can pick up a bike from one of the many docking stations located throughout the city, ride to their destination, ' 
        'and then return the bike to any available docking station. This initiative aims to promote sustainable and accessible transportation alternatives the Boston metro area.')
        
        st.markdown(bluebike_description)
        
        # Create a map with all the bike stations
        station_map(df_stations)
        
        stations_number = df_stations.shape[0]
        st.markdown(f'The {stations_number} different stations located around Boston mostly reside in the Cambridge area.'
                    ' The Blue Bike stations follow along the Charles River, starting at around Charlestown/North End '
                    'and continue all the way down to around Harvard University.')

    with tab2:
        st.header(':blue[Plan Your Next Trip]')
        st.image(bluebikes_image2)
        st.markdown('Using our trip planner tool, you can easily calculate how far your next commute will be along'
                     ' with an accurate estimated time of arrival (ETA) based on user data')
        
        start = st.selectbox('Start location',df_stations['station_name'])
        end = st.selectbox('End location',df_stations['station_name'])

        trip_map(start,end,df_stations)

        trip_length = trip_distance(start,end,df_stations)


        trip_b0,trip_b1 = simple_regression(x=df_trips['distance (mi)'],y=df_trips['duration (min)'])
        
        if trip_length == 0:
            eta = 0
        else:
            eta = trip_b0 + trip_b1 * trip_length

        st.markdown(f'Your trip is {trip_length:.3f} miles long')
        st.markdown(f'Your ETA is {eta:.2f} minutes')
        


    with tab3:
        st.header(':blue[Blue Bike User Data]')
        st.subheader(':blue[What times are the most popular for people to use Blue Bikes?]')
        
        common_hours = df_trips.sort_values(by=['hour'])
        plt.figure(figsize=(10,6))
        plt.hist(common_hours['hour'],bins=24,color='#004c6d',edgecolor='black',alpha=0.7)
        plt.title('Most Common Time for Bike Trips')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Bike Trips')
        plt.xticks(range(24))

        most_common_hour = common_hours['hour'].mode().values[0]
        plt.axvline(most_common_hour, color='black', linestyle='dashed',
                     linewidth=2, label=f'Most Common Hour: {most_common_hour}')
        plt.legend()
        st.pyplot(plt.gcf())
        st.markdown('This plot shows the distribution of Blue Bike use by hour of the day.'
                    ' During the month of January, 2015, 8am was the most common time for people to use Blue Bikes.'
                    ' There\'s another peak at 5pm. '
                    'This supports the hypothesis that most Blue Bikes users ride to commute'
                    ' to work/school.')
        plt.clf()        

        st.subheader(':blue[What usage differences are there between one-time customers and Blue Bike subscribers?]')
        fig, axs= plt.subplots(3,1,figsize=(5,5))

        counts = df_trips.groupby(['usertype']).size().reset_index(name='count')
        usertype = df_trips[['usertype','duration (min)','distance (mi)']].groupby(['usertype']).sum(numeric_only=True).reset_index()
        colors = ['#004c6d','#c1e7ff']

        axs[0].pie(usertype['duration (min)'], labels=counts['usertype'], autopct='%1.1f%%',
                startangle=90, colors=colors)
        axs[0].set_title('Distribution of Total Ride Time (in minutes) by User Type')

        axs[1].pie(usertype['distance (mi)'], labels=counts['usertype'], autopct='%1.1f%%', startangle=90,colors=colors)
        axs[1].set_title('Distribution of Total Ride Distance fby User Type') 

        axs[2].pie(counts['count'], labels=counts['usertype'], autopct='%1.1f%%', startangle=90,colors=colors)
        axs[2].set_title('Distribution of User Type')

        st.markdown('Blue Bikes have two different pricing plans, the single trip (customers) and their membership plan.'
                    ' It would be interesting to look at if members make up a disproportionate amount of bike use.'
                    ' Similar to gym plans, bike members could make a smaller proportion of bike use,'
                    ' or members could be making the most of their money through putting their membership to good use.')
        st.pyplot(fig)
        plt.clf()
        st.markdown("These pie charts show that subscribers make up almost exactly proportionate amounts of bike use compared to how many subscribers there are.")
        
        st.subheader(':blue[At what speed do Blue Bike users ride?]')
        fig, ax = plt.subplots(figsize=(6,6))

        threshold = 40 # This subsets the data to trips less than 40 minutes long. 
                        # A few outlier (two+ hour long trips) mess up the scale of the graph
        speed_data = df_trips[df_trips['duration (min)'] < threshold]

        ax.scatter(speed_data['duration (min)'],speed_data['distance (mi)'])
        plt.xlabel('Time (in minutes)')
        plt.ylabel('Distance Traveled (in miles)')
        plt.title('Time Users Spend vs Distance Traveled')
        st.pyplot(fig)
        st.markdown('This scatterplot shows the amount of time users spend on a ride, along with how far they go.'
                    ' There is a general linear relationship, where the more time people spend on their trip,'
                    ' generally they travel a farther difference. Looking at user data, Blue Bike users generally'
                    f' ride at about {trip_b1:.2f} miles per hour')

if __name__ == "__main__":
    main()
