##################################################################################### Load dependencies #####################################################################################

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import json



##################################################################################### Load and preprocess data #####################################################################################

# Data sets on global surface temperature development
df_temperature_global_kag = pd.read_csv('../Data/GlobalTemperatures.csv')

df_temperature_by_country_owd = pd.read_csv('../Data/average-monthly-surface-temperature.csv')



# 1. Plot: Data preprocessing
df_temperature_global_kag['Year'] = df_temperature_global_kag['dt'].str[0:4]
df_temperature_global_kag['LandAverageTemperature+Unc'] = df_temperature_global_kag['LandAverageTemperature']+df_temperature_global_kag['LandAverageTemperatureUncertainty']
df_temperature_global_kag['LandAverageTemperature-Unc'] = df_temperature_global_kag['LandAverageTemperature']-df_temperature_global_kag['LandAverageTemperatureUncertainty']
df_temperature_global_kag['LandMaxTemperature+Unc'] = df_temperature_global_kag['LandMaxTemperature']+df_temperature_global_kag['LandMaxTemperatureUncertainty']
df_temperature_global_kag['LandMaxTemperature-Unc'] = df_temperature_global_kag['LandMaxTemperature']-df_temperature_global_kag['LandMaxTemperatureUncertainty']
df_temperature_global_kag['LandMinTemperature+Unc'] = df_temperature_global_kag['LandMinTemperature']+df_temperature_global_kag['LandMinTemperatureUncertainty']
df_temperature_global_kag['LandMinTemperature-Unc'] = df_temperature_global_kag['LandMinTemperature']-df_temperature_global_kag['LandMinTemperatureUncertainty']

df_temperature_global_kag_ocean = df_temperature_global_kag[df_temperature_global_kag["Year"] >= "1850"].copy()
df_temperature_global_kag_ocean['LandAndOceanAverageTemperature+Unc'] = df_temperature_global_kag['LandAndOceanAverageTemperature'][df_temperature_global_kag["Year"] >= "1850"]+df_temperature_global_kag['LandAndOceanAverageTemperatureUncertainty'][df_temperature_global_kag["Year"] >= "1850"]
df_temperature_global_kag_ocean['LandAndOceanAverageTemperature-Unc'] = df_temperature_global_kag['LandAndOceanAverageTemperature'][df_temperature_global_kag["Year"] >= "1850"]-df_temperature_global_kag['LandAndOceanAverageTemperatureUncertainty'][df_temperature_global_kag["Year"] >= "1850"]


# Select data for the top and bottom lines
top_line = df_temperature_global_kag_ocean.groupby(['Year'])['LandAndOceanAverageTemperature+Unc'].mean()
bottom_line = df_temperature_global_kag_ocean.groupby(['Year'])['LandAndOceanAverageTemperature-Unc'].mean()
average_line = df_temperature_global_kag_ocean.groupby(['Year'])['LandAndOceanAverageTemperature'].mean()

# Calculate the mean temperature of 1850 to 1900 plus 1.5 degrees Celsius
mean_1850_1900 = df_temperature_global_kag_ocean[(df_temperature_global_kag_ocean['Year'] >= '1850') & (df_temperature_global_kag_ocean['Year'] <= '1900')]['LandAndOceanAverageTemperature'].mean()
mean_1850_1900_plus_1_5 = mean_1850_1900 + 1.5

# Perform polynomial regression
x = np.arange(len(average_line.index)).reshape(-1, 1)
y = average_line.values.reshape(-1, 1)

degree = 2
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)

x_future = np.arange(len(average_line.index), len(average_line.index) + 86).reshape(-1, 1)
x_future_poly = poly_features.transform(x_future)
y_future_pred = model.predict(x_future_poly)

last_two_points_x = x[-2:].flatten()
last_two_points_y = y_pred[-2:].flatten()

tangent_slope, tangent_intercept = np.polyfit(last_two_points_x, last_two_points_y, 1)

last_observed_y = y_pred[-1][0]

tangent_intercept = last_observed_y - tangent_slope * 2015

x_tangent_extension = np.arange(2015, 2101).reshape(-1, 1)
y_tangent_extension = tangent_slope * x_tangent_extension + tangent_intercept

difference_future = y_tangent_extension.flatten() - (y_future_pred.flatten() - y_tangent_extension.flatten())


# Define line graphs
g2_graph_1 = go.Scatter(x=top_line.index, y=top_line, mode="lines", name="Durchschnittliche Oberflächentemperatur bei maximaler Messunsicherheit nach oben", line=dict(color="orange"),
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Durchschnittliche Oberflächentemperatur bei</br>maximaler Messunsicherheit nach oben</b>: %{y}ºC<extra></extra>")
g2_graph_2 = go.Scatter(x=bottom_line.index, y=bottom_line, mode="lines", name="Durchschnittliche Oberflächentemperatur bei maximaler Messunsicherheit nach unten", line=dict(color="orange"), fill="tonexty", fillcolor="rgba(255,165,0,0.4)",
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Durchschnittliche Oberflächentemperatur bei</br>maximaler Messunsicherheit nach unten</b>: %{y}ºC<extra></extra>")
g2_graph_3 = go.Scatter(x=average_line.index, y=average_line, mode="lines", name="Durchschnittliche Oberflächentemperatur", line=dict(color="red"),
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Durchschnittliche Oberflächentemperatur</b>: %{y}ºC<extra></extra>")
g2_graph_4 = go.Scatter(x=[str(i) for i in list((range(1850, 2100)))], y=[mean_1850_1900_plus_1_5 for _ in range(len(list((range(1850, 2100)))))], mode="lines", name="1,5-Grad-Ziel des Pariser Klimaabkommens für 2100", line=dict(color="grey"),
                        hovertemplate="<br><b>1,5-Grad-Ziel des Pariser Klimaabkommens für 2100</b><extra></extra>")
g2_graph_5 = go.Scatter(x=average_line.index, y=y_pred.flatten(), mode="lines", name="Annäherung des historischen Temperaturverlaufes durch Polynomregression", line=dict(color="black"),
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Historische Annäherung durch Polynomregression</b>: %{y}ºC<extra></extra>")
g2_graph_6 = go.Scatter(x=np.arange(2015, 2101), y=y_future_pred.flatten(), mode="lines", name="Temperaturprognose bei anhaltendem Trend", line=dict(color="red"),
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Temperaturprognose bei anhaltendem Trend</b>: %{y}ºC<extra></extra>")
g2_graph_7 = go.Scatter(x=np.arange(2015, 2101), y=difference_future, mode="lines", name="Temperaturprognose bei abflachendem Trend", line=dict(color="green"), fill='tonexty', fillcolor='rgba(211, 211, 211, 0.4)',
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Temperaturprognose bei abflachendem Trend</b>: %{y}ºC<extra></extra>")
g2_graph_8 = go.Scatter(x=x_tangent_extension.flatten(), y=y_tangent_extension.flatten(), mode="lines", name="Temperaturprognose bei verfestigtem Trend", line=dict(color="black"),
                        hovertemplate="<b>Jahr</b>: %{x}<br><b>Temperaturprognose bei verfestigtem Trend</b>: %{y}ºC<extra></extra>")



# 2. Plot: Preprocess data
df_temperature_by_country_owd = df_temperature_by_country_owd.drop(["Code", "Day", "Average surface temperature"], axis=1).rename(columns={"Entity": "Country", "year": "Year", "Average surface temperature.1": "AverageSurfaceTemperature"}).groupby(["Country", "Year"]).mean().reset_index()


# Load geo JSON
with open('../Data/countries.geo.json') as f:
    world_json = json.load(f)


# Check if the geo JSON contains different country names than the dataframe
countries_world_json = tuple(feature['properties']['name'] for feature in world_json['features'])
countries_data = df_temperature_by_country_owd["Country"].unique()

diff1 = [x for x in countries_world_json if x not in countries_data]
diff2 = [x for x in countries_data if x not in countries_world_json]


# Adapt information in geo JSON to ensure compatibility
dict_countries = {
    'United States of America': 'United States',
    'Republic of Serbia': 'Serbia',
    'The Bahamas': 'Bahamas',
    'Trinidad and Tobago': 'Trinidad And Tobago',
    'Saint Vincent and the Grenadines': 'Saint Vincent And The Grenadines',
    'Antigua and Barbuda': 'Antigua And Barbuda',
    'Saint Kitts and Nevis': 'Saint Kitts And Nevis',
    'United Republic of Tanzania': 'Tanzania',
    'Republic of the Congo': 'Congo',
    'Democratic Republic of the Congo': 'Democratic Republic of Congo',
    'Ivory Coast': "Côte D'Ivoire",
    'eSwatini': 'Swaziland',
    'Guinea Bissau': 'Guinea-Bissau',
    'São Tomé and Principe': 'Sao Tome And Principe',
    'Cabo Verde': 'Cape Verde',
    'Czechia': 'Czech Republic',
    'North Macedonia': 'Macedonia',
    'Republic of Serbia': 'Serbia',
    'Bosnia and Herzegovina': 'Bosnia And Herzegovina',
    'Turks and Caicos Islands': 'Turks And Caicas Islands',
    'Saint Pierre and Miquelon': 'Saint Pierre And Miquelon',
    'United States Virgin Islands': 'Virgin Islands',
    'Saint Barthelemy': 'Saint Barthélemy',
    'Falkland Islands': 'Falkland Islands (Islas Malvinas)',
    'Palestine': 'Palestina',
    'East Timor': 'Timor Leste',
    'Hong Kong S.A.R.': 'Hong Kong',
    'Macao S.A.R': 'Macau',
    'Isle of Man': 'Isle Of Man',
    'Aland': 'Åland',
    'Federated States of Micronesia': 'Federated States Of Micronesia'
}

for feature in world_json['features']:
    country_name = feature['properties']['name']
    if country_name in dict_countries:
        feature['properties']['name'] = dict_countries[country_name]


# Calculate mean temperature for 1940-1959
df_mean_1940_1959 = df_temperature_by_country_owd.loc[(df_temperature_by_country_owd["Year"] >= 1940) & (df_temperature_by_country_owd["Year"] < 1960), ["Year", "Country", "AverageSurfaceTemperature"]].groupby("Country")["AverageSurfaceTemperature"].mean().to_frame().rename(columns={"AverageSurfaceTemperature": "AverageTemperature_40_59"})
df_interactive = df_temperature_by_country_owd.merge(df_mean_1940_1959, on="Country", how="left")

df_interactive["AverageTemperatureDifference"] = df_interactive["AverageSurfaceTemperature"] - df_interactive["AverageTemperature_40_59"]



# 3. Plot: Data sets on emissions development
df_emissions_by_country_owd = pd.read_csv('../Data/co-emissions-per-capita.csv')

df_total_emissions_by_sectors_owd = pd.read_csv('../Data/ghg-emissions-by-sector-stacked.csv')
df_co2_emissions_by_country_owd = pd.read_csv('../Data/co-emissions-by-sector.csv')
df_nitrous_oxide_emissions_by_country_owd = pd.read_csv('../Data/nitrous-oxide-emissions-by-sector.csv')
df_methane_emissions_by_country_owd = pd.read_csv('../Data/methane-emissions-by-sector.csv')


# Preprocess data
df_emissions_by_country_owd = df_emissions_by_country_owd.drop(["Code"], axis=1).rename(columns={"Entity": "Country", "year": "Year"}).groupby(["Country", "Year"]).mean().reset_index()



# 4. Plot: Preprocess data
df_total_emissions_by_sectors_owd = df_total_emissions_by_sectors_owd.drop(["Code"], axis=1).rename(columns={"Entity": "Country", "Greenhouse gas emissions from other fuel combustion": "Verbrennung anderer Brennstoffe", "Greenhouse gas emissions from bunker fuels": "Schweröl", "Greenhouse gas emissions from waste": "Abfälle", "Greenhouse gas emissions from buildings": "Gebäude", "Greenhouse gas emissions from industry": "Industrie", "Fugitive emissions of greenhouse gases from energy production": "Diffuse Emissionen", "Greenhouse gas emissions from agriculture": "Landwirtschaft", "Greenhouse gas emissions from manufacturing and construction": "Herstellung und Bau", "Greenhouse gas emissions from transport": "Transport", "Greenhouse gas emissions from electricity and heat": "Elektrizität und Wärme"})
df_co2_emissions_by_country_owd = df_co2_emissions_by_country_owd.drop(["Code"], axis=1).rename(columns={"Entity": "Country", "Carbon dioxide emissions from buildings": "Gebäude", "Carbon dioxide emissions from industry": "Industrie", "Carbon dioxide emissions from land use change and forestry": "Flächennutzung und Forstwirtschaft", "Carbon dioxide emissions from other fuel combustion": "Verbrennung anderer Brennstoffe", "Carbon dioxide emissions from transport": "Transport", "Carbon dioxide emissions from manufacturing and construction": "Herstellung und Bau", "Fugitive emissions of carbon dioxide from energy production": "Diffuse Emissionen", "Carbon dioxide emissions from electricity and heat": "Elektrizität und Wärme"})
df_nitrous_oxide_emissions_by_country_owd = df_nitrous_oxide_emissions_by_country_owd.drop(["Code"], axis=1).rename(columns={"Entity": "Country", "Nitrous oxide emissions from agriculture": "Landwirtschaft", "Nitrous oxide emissions from industry": "Industrie", "Nitrous oxide emissions from other fuel combustion": "Verbrennung anderer Brennstoffe", "Nitrous oxide emissions from waste": "Abfälle", "Nitrous oxide emissions from land use change and forestry": "Flächennutzung und Forstwirtschaft", "Fugitive emissions of nitrous oxide from energy production": "Energieerzeugung"})
df_methane_emissions_by_country_owd = df_methane_emissions_by_country_owd.drop(["Code"], axis=1).rename(columns={"Entity": "Country", "Methane emissions from agriculture": "Landwirtschaft", "Fugitive emissions of methane from energy production": "Diffuse Emissionen", "Methane emissions from waste": "Abfälle", "Methane emissions from land use change and forestry": "Flächennutzung und Forstwirtschaft", "Methane emissions from other fuel combustion": "Verbrennung anderer Brennstoffe", "Methane emissions from industry": "Industrie"})


# Add a total value per year
df_total_emissions_by_sectors_owd['Total'] = df_total_emissions_by_sectors_owd[['Verbrennung anderer Brennstoffe', 'Schweröl',
                                                                                'Abfälle', 'Gebäude', 'Industrie', 'Diffuse Emissionen',
                                                                                'Landwirtschaft', 'Herstellung und Bau', 'Transport',
                                                                                'Elektrizität und Wärme']].sum(axis=1)

df_co2_emissions_by_country_owd['Total'] = df_co2_emissions_by_country_owd[['Gebäude', 'Industrie',
                                                                            'Flächennutzung und Forstwirtschaft', 'Verbrennung anderer Brennstoffe',
                                                                            'Transport', 'Herstellung und Bau', 'Diffuse Emissionen',
                                                                            'Elektrizität und Wärme']].sum(axis=1)

df_nitrous_oxide_emissions_by_country_owd['Total'] = df_nitrous_oxide_emissions_by_country_owd[['Landwirtschaft', 'Industrie',
                                                                                                'Verbrennung anderer Brennstoffe', 'Abfälle',
                                                                                                'Flächennutzung und Forstwirtschaft', 'Energieerzeugung']].sum(axis=1)

df_methane_emissions_by_country_owd['Total'] = df_methane_emissions_by_country_owd[['Landwirtschaft', 'Diffuse Emissionen', 'Abfälle',
                                                                                    'Flächennutzung und Forstwirtschaft', 'Verbrennung anderer Brennstoffe',
                                                                                    'Industrie']].sum(axis=1)



# 5. Plot: Preprocess data
df_threatened_species_owd = pd.read_csv('../Data/global-living-planet-index.csv')
df_natural_disasters_owd = pd.read_csv('../Data/natural-disasters-by-type.csv')
df_ice_sheet_mass_owd = pd.read_csv('../Data/ice-sheet-mass-balance.csv')
df_sea_level_owd = pd.read_csv('../Data/sea-level.csv')

# Subplot 5.1: Threatened species
df_threatened_species_owd = df_threatened_species_owd[df_threatened_species_owd['Entity']=='World'].drop(['Code', 'Entity'], axis=1)


# Create the line plot
fig_7 = go.Figure()

# Add the Living Planet Index line
fig_7.add_trace(go.Scatter(x=df_threatened_species_owd['Year'], y=df_threatened_species_owd['Living Planet Index'], mode='lines', name="Anzahl dokumentierter Wildtierpopulationen ggü. 1970", line=dict(color="green"),
                           hovertemplate="<b>Jahr</b>: %{x}<br><b>Anzahl dokumentierter Wildtierpopulationen ggü. 1970</b>: %{y}%<extra></extra>"))

# Add the upper and lower confidence intervals
fig_7.add_trace(go.Scatter(x=df_threatened_species_owd['Year'], y=df_threatened_species_owd['Upper CI'], mode='lines', line=dict(color='lightgreen'), name="Oberes Konfidenzintervall",
                           hovertemplate="<b>Jahr</b>: %{x}<br><b>Oberes Konfidenzintervall</b>: %{y}%<extra></extra>"))
fig_7.add_trace(go.Scatter(x=df_threatened_species_owd['Year'], y=df_threatened_species_owd['Lower CI'], mode='lines', line=dict(color='lightgreen'), name="Unteres Konfidenzintervall",
                           hovertemplate="<b>Jahr</b>: %{x}<br><b>Unteres Konfidenzintervall</b>: %{y}%<extra></extra>", fill='tonexty'))

# Update layout
fig_7.update_layout(title={'text': 'Living Planet Index: Anzahl dokumentierter Wildtierpopulationen ggü. 1970', 'x':0.5, 'xanchor':'center'},
                    xaxis=dict(
                        title="Jahr",
                        tickmode="linear",
                        dtick=5,
                        tickangle=0,
                        tickfont=dict(size=12),
                        ticks="outside",
                        showline=True,
                        linecolor="black",
                    ),
                    yaxis=dict(
                        title="Living Planet Index",
                        ticks="outside",
                        showline=True,
                        linecolor="black",
                        gridcolor="lightgrey",
                        zeroline=True,
                        zerolinecolor="lightgrey",
                        zerolinewidth=1,
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='closest')


# Subplot 5.2: Natural disasters
df_natural_disasters_owd = df_natural_disasters_owd.drop(["Code"], axis=1)
df_natural_disasters_owd = df_natural_disasters_owd[df_natural_disasters_owd['Year'] != 2024]
df_natural_disasters_owd = df_natural_disasters_owd[df_natural_disasters_owd['Entity'] != 'All disasters']
df_natural_disasters_owd['Entity'] = df_natural_disasters_owd['Entity'].str.replace('Drought', 'Dürre').replace('Wildfire', 'Waldbrand').replace('Flood', 'Flut').replace('Extreme weather', 'Extremwetter').replace('Volcanic activity', 'Vulkanische Aktivität').replace('Earthquake', 'Erdbeben').replace('Extreme temperature', 'Extremtemperatur').replace('Wet mass movement', 'Nassmassenbewegung').replace('Dry mass movement', 'Trockenmassenbewegung').replace('Glacial lake outburst flood', 'Gletschersee-Ausbruchshochwasser')


# Define the number of entities
num_entities_g8 = 10

# Get colors from Viridis color scale
colors_viridis = px.colors.sequential.turbid[:num_entities_g8]

# Create the traces for the bar chart
traces_g8 = []
for i, (entity, group) in enumerate(df_natural_disasters_owd.groupby(['Year', 'Entity']).sum().reset_index().groupby('Entity')):
    trace_g8 = go.Bar(
        x=group['Year'],
        y=group['Number of reported natural disasters'],
        name=entity,
        hovertemplate='<b>' + f'{entity}' + '</b><br>Anzahl dokumentierter Ereignisse im Jahr %{x}: %{y}<extra></extra>',
        marker=dict(color=colors_viridis[i])
    )
    traces_g8.append(trace_g8)

layout_g8 = go.Layout(
    title={'text': 'Anzahl global dokumentierter Naturkatastrophen', 'x':0.5, 'xanchor':'center'},

    xaxis=dict(
        title="Jahr",
        tickmode="linear",
        dtick=5,
        tickangle=0,
        tickfont=dict(size=12),
        ticks="outside",
        showline=True,
        linecolor="black",
    ),
    yaxis=dict(
        title="Anzahl dokumentierter Naturkatastrophen",
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        zeroline=True,
        zerolinecolor="lightgrey",
        zerolinewidth=1,
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    barmode='stack',
    hovermode='closest'
)

# Create the figure
fig_8 = go.Figure(data=traces_g8, layout=layout_g8)



# Subplot 5.3: Ice sheet mass development
df_ice_sheet_mass_owd = df_ice_sheet_mass_owd.drop(['Code'], axis=1)
df_ice_sheet_mass_owd['Day'] = pd.to_datetime(df_ice_sheet_mass_owd['Day'])
df_ice_sheet_mass_owd['Year'] = df_ice_sheet_mass_owd['Day'].dt.year


# Create the line plot
fig_9 = go.Figure()

# Add the two lines for Antarctic and Greenland
fig_9.add_trace(go.Scatter(x=df_ice_sheet_mass_owd['Day'].unique(), y=df_ice_sheet_mass_owd['Cumulative change in mass in the ice sheets, according to NASA/JPL'][df_ice_sheet_mass_owd['Entity'] == 'Greenland'], mode='lines', name='Grönland', line=dict(color='rgb(177, 143, 219)'),
                           hovertemplate="<b>Datum</b>: %{x}<br><b>Absolute Veränderung der Masse der<br>Eisflächen Grönlands ggü. 1970</b>: %{y} Mrd. t<extra></extra>"))

fig_9.add_trace(go.Scatter(x=df_ice_sheet_mass_owd['Day'].unique(), y=df_ice_sheet_mass_owd['Cumulative change in mass in the ice sheets, according to NASA/JPL'][df_ice_sheet_mass_owd['Entity'] == 'Antarctica'], mode='lines', name='Antarktis', line=dict(color='purple'),
                           hovertemplate="<b>Datum</b>: %{x}<br><b>Absolute Veränderung der Masse der<br>Eisflächen der Antarktis ggü. 1970</b>: %{y} Mrd. t<extra></extra>"))


# Update layout
fig_9.update_layout(title={'text': 'Absolute Veränderung der Masse der Eisflächen Grönlands und der Antarktis (in Mrd. Tonnen)', 'x':0.5, 'xanchor':'center'},
                    xaxis=dict(
                        title="Jahr",
                        tickmode="array",
                        tickvals=df_ice_sheet_mass_owd['Day'].unique()[::10],
                        ticktext=df_ice_sheet_mass_owd['Day'].unique()[::10].strftime('%Y'),
                        dtick=200000,
                        tickangle=0,
                        tickfont=dict(size=12),
                        ticks="outside",
                        showline=True,
                        linecolor="black",
                    ),
                    yaxis=dict(
                        title="Eisflächenmassenveränderung (in Mrd. t)",
                        ticks="outside",
                        showline=True,
                        linecolor="black",
                        gridcolor="lightgrey",
                        zeroline=True,
                        zerolinecolor="lightgrey",
                        zerolinewidth=1,
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='closest')



# Subplot 5.4: Sea level development
df_sea_level_owd = df_sea_level_owd.drop(['Code'], axis=1)
df_sea_level_owd['Day'] = pd.to_datetime(df_sea_level_owd['Day'])
df_sea_level_owd['Year'] = df_sea_level_owd['Day'].dt.year


# Create the line plot
fig_10 = go.Figure()

# Add the two lines for Antarctic and Greenland
fig_10.add_trace(go.Scatter(x=df_sea_level_owd['Day'].unique(), y=df_sea_level_owd['Global sea level as an average of Church and White (2011) and UHSLC data'], mode='lines', name='Meeresspiegel', line=dict(color='rgb(90, 140, 219)'),
                            hovertemplate="<b>Datum</b>: %{x}<br><b>Meeresspiegel im Vergleich zum Durchschnitt der Jahre 1993 - 2008</b>: %{y} mm<extra></extra>"))

# Update layout
fig_10.update_layout(title={'text': 'Globaler durchschnittlicher Meeresspiegel im absoluten<br>Vergleich zum Durchschnitt der Jahre 1993 - 2008 (in mm)', 'x':0.5, 'xanchor':'center'},
                     xaxis=dict(
                         title="Jahr",
                         tickmode="array",
                         tickvals=df_sea_level_owd['Day'].unique()[::20],
                         ticktext=df_sea_level_owd['Day'].unique()[::20].strftime('%Y'),
                         dtick=200000,
                         tickangle=0,
                         tickfont=dict(size=12),
                         ticks="outside",
                         showline=True,
                         linecolor="black",
                     ),
                     yaxis=dict(
                         title="Meeresspiegel (in mm)",
                         ticks="outside",
                         showline=True,
                         linecolor="black",
                         gridcolor="lightgrey",
                         zeroline=True,
                         zerolinecolor="lightgrey",
                         zerolinewidth=1,
                     ),
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)',
                     hovermode='closest')



# 7. Plot: Reload temperature data
df_parallel = df_temperature_by_country_owd

df_parallel_mean_1940_1959 = df_parallel.loc[
    (df_parallel["Year"] >= 1940) & (df_parallel["Year"] < 1960),
    ["Year", "Country", "AverageSurfaceTemperature"]
].groupby("Country")["AverageSurfaceTemperature"].mean().to_frame().rename(columns={"AverageSurfaceTemperature": "AverageTemperature_40_59"})
df_parallel = df_parallel.merge(df_parallel_mean_1940_1959, on="Country", how="left")
df_parallel["AverageTemperatureDifference"] = df_parallel["AverageSurfaceTemperature"] - df_parallel["AverageTemperature_40_59"]


# Calculate difference between mean of 1940 - 1960 and mean of 2013 to 2023
df_parallel = df_parallel.loc[(df_parallel['Year'] >= 2013) & (df_parallel['Year'] <= 2023) & (df_parallel['Country'] != 'World')].reset_index()
df_parallel = df_parallel.groupby('Country')[['AverageTemperatureDifference', 'AverageSurfaceTemperature']].mean().reset_index()


# Reload emission data
df_parallel_total_emissions_by_sectors_owd = df_total_emissions_by_sectors_owd.groupby('Country')['Total'].mean().to_frame().reset_index()

# Merge dataframes
df_parallel = df_parallel.merge(df_parallel_total_emissions_by_sectors_owd, on='Country', how='left')


# Map countries to continents
country_to_continent = {
    'Afghanistan': 'Asia',
    'Albania': 'Europe',
    'Algeria': 'Africa',
    'American Samoa': 'Oceania',
    'Andorra': 'Europe',
    'Angola': 'Africa',
    'Anguilla': 'North America',
    'Antigua and Barbuda': 'North America',
    'Argentina': 'South America',
    'Armenia': 'Asia',
    'Australia': 'Oceania',
    'Austria': 'Europe',
    'Azerbaijan': 'Asia',
    'Bahamas': 'North America',
    'Bahrain': 'Asia',
    'Bangladesh': 'Asia',
    'Barbados': 'North America',
    'Belarus': 'Europe',
    'Belgium': 'Europe',
    'Belize': 'North America',
    'Benin': 'Africa',
    'Bhutan': 'Asia',
    'Bolivia': 'South America',
    'Bosnia and Herzegovina': 'Europe',
    'Botswana': 'Africa',
    'Brazil': 'South America',
    'Brunei': 'Asia',
    'Bulgaria': 'Europe',
    'Burkina Faso': 'Africa',
    'Burundi': 'Africa',
    'Cambodia': 'Asia',
    'Cameroon': 'Africa',
    'Canada': 'North America',
    'Cape Verde': 'Africa',
    'Cayman Islands': 'North America',
    'Central African Republic': 'Africa',
    'Chad': 'Africa',
    'Chile': 'South America',
    'China': 'Asia',
    'Colombia': 'South America',
    'Comoros': 'Africa',
    'Congo': 'Africa',
    'Cook Islands': 'Oceania',
    'Costa Rica': 'North America',
    'Cote d\'Ivoire': 'Africa',
    'Croatia': 'Europe',
    'Cuba': 'North America',
    'Cyprus': 'Asia',
    'Czechia': 'Europe',
    'Democratic Republic of Congo': 'Africa',
    'Denmark': 'Europe',
    'Djibouti': 'Africa',
    'Dominica': 'North America',
    'Dominican Republic': 'North America',
    'East Timor': 'Asia',
    'Ecuador': 'South America',
    'Egypt': 'Africa',
    'El Salvador': 'North America',
    'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa',
    'Estonia': 'Europe',
    'Eswatini': 'Africa',
    'Ethiopia': 'Africa',
    'Falkland Islands': 'South America',
    'Faroe Islands': 'Europe',
    'Fiji': 'Oceania',
    'Finland': 'Europe',
    'France': 'Europe',
    'French Polynesia': 'Oceania',
    'Gabon': 'Africa',
    'Gambia': 'Africa',
    'Georgia': 'Asia',
    'Germany': 'Europe',
    'Ghana': 'Africa',
    'Greece': 'Europe',
    'Greenland': 'North America',
    'Grenada': 'North America',
    'Guatemala': 'North America',
    'Guinea': 'Africa',
    'Guinea-Bissau': 'Africa',
    'Guyana': 'South America',
    'Haiti': 'North America',
    'Heard Island and McDonald Islands': 'Antarctica',
    'Honduras': 'North America',
    'Hong Kong': 'Asia',
    'Hungary': 'Europe',
    'Iceland': 'Europe',
    'India': 'Asia',
    'Indonesia': 'Asia',
    'Iran': 'Asia',
    'Iraq': 'Asia',
    'Ireland': 'Europe',
    'Israel': 'Asia',
    'Italy': 'Europe',
    'Jamaica': 'North America',
    'Japan': 'Asia',
    'Jordan': 'Asia',
    'Kazakhstan': 'Asia',
    'Kenya': 'Africa',
    'Kiribati': 'Oceania',
    'Kosovo': 'Europe',
    'Kuwait': 'Asia',
    'Kyrgyzstan': 'Asia',
    'Laos': 'Asia',
    'Latvia': 'Europe',
    'Lebanon': 'Asia',
    'Lesotho': 'Africa',
    'Liberia': 'Africa',
    'Libya': 'Africa',
    'Liechtenstein': 'Europe',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Madagascar': 'Africa',
    'Malawi': 'Africa',
    'Malaysia': 'Asia',
    'Maldives': 'Asia',
    'Mali': 'Africa',
    'Malta': 'Europe',
    'Marshall Islands': 'Oceania',
    'Mauritania': 'Africa',
    'Mauritius': 'Africa',
    'Mexico': 'North America',
    'Micronesia': 'Oceania',
    'Moldova': 'Europe',
    'Monaco': 'Europe',
    'Mongolia': 'Asia',
    'Montenegro': 'Europe',
    'Morocco': 'Africa',
    'Mozambique': 'Africa',
    'Myanmar': 'Asia',
    'Namibia': 'Africa',
    'Nauru': 'Oceania',
    'Nepal': 'Asia',
    'Netherlands': 'Europe',
    'New Caledonia': 'Oceania',
    'New Zealand': 'Oceania',
    'Nicaragua': 'North America',
    'Niger': 'Africa',
    'Nigeria': 'Africa',
    'Niue': 'Oceania',
    'North Korea': 'Asia',
    'North Macedonia': 'Europe',
    'Northern Mariana Islands': 'Oceania',
    'Norway': 'Europe',
    'Oman': 'Asia',
    'Pakistan': 'Asia',
    'Palau': 'Oceania',
    'Palestine': 'Asia',
    'Panama': 'North America',
    'Papua New Guinea': 'Oceania',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Philippines': 'Asia',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Puerto Rico': 'North America',
    'Qatar': 'Asia',
    'Romania': 'Europe',
    'Russia': 'Europe',
    'Rwanda': 'Africa',
    'Saint Helena': 'Africa',
    'Saint Kitts and Nevis': 'North America',
    'Saint Lucia': 'North America',
    'Saint Vincent and the Grenadines': 'North America',
    'Samoa': 'Oceania',
    'San Marino': 'Europe',
    'Sao Tome and Principe': 'Africa',
    'Saudi Arabia': 'Asia',
    'Senegal': 'Africa',
    'Serbia': 'Europe',
    'Seychelles': 'Africa',
    'Sierra Leone': 'Africa',
    'Singapore': 'Asia',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'Solomon Islands': 'Oceania',
    'Somalia': 'Africa',
    'South Africa': 'Africa',
    'South Korea': 'Asia',
    'South Sudan': 'Africa',
    'Spain': 'Europe',
    'Sri Lanka': 'Asia',
    'Sudan': 'Africa',
    'Suriname': 'South America',
    'Sweden': 'Europe',
    'Switzerland': 'Europe',
    'Syria': 'Asia',
    'Taiwan': 'Asia',
    'Tajikistan': 'Asia',
    'Tanzania': 'Africa',
    'Thailand': 'Asia',
    'Togo': 'Africa',
    'Tonga': 'Oceania',
    'Trinidad and Tobago': 'North America',
    'Tunisia': 'Africa',
    'Turkey': 'Asia',
    'Turkmenistan': 'Asia',
    'Tuvalu': 'Oceania',
    'Uganda': 'Africa',
    'Ukraine': 'Europe',
    'United Arab Emirates': 'Asia',
    'United Kingdom': 'Europe',
    'United States': 'North America',
    'Uruguay': 'South America',
    'Uzbekistan': 'Asia',
    'Vanuatu': 'Oceania',
    'Vatican City': 'Europe',
    'Venezuela': 'South America',
    'Vietnam': 'Asia',
    'Yemen': 'Asia',
    'Zambia': 'Africa',
    'Zimbabwe': 'Africa'
}


# Add column with continent information
df_parallel['Continent'] = df_parallel['Country'].map(country_to_continent)


# Create a mapping of continents to numerical labels
continent_labels = {'Africa': 1, 'Asia': 2, 'Europe': 3, 'North America': 4, 'Oceania': 5, 'South America': 6}

# Map continent names to numerical labels
df_parallel_num = df_parallel.copy()
df_parallel_num['Continent'] = df_parallel['Continent'].map(continent_labels)

# Reverse the continent labels mapping for hoverinfo
continent_labels_reverse = {v: k for k, v in continent_labels.items()}

# Create the parallel coordinates plot
fig_11 = go.Figure(data=go.Parcoords(
    line=dict(color=df_parallel_num['Continent'], colorscale='Plasma', showscale=False),
    dimensions=[
        dict(label='Kontinent', values=df_parallel_num['Continent'], tickvals=list(continent_labels.values()), ticktext=list(continent_labels_reverse.values())),
        dict(range=[df_parallel_num['AverageTemperatureDifference'].min(), df_parallel_num['AverageTemperatureDifference'].max()],
             label='Durchschnittl. Temperaturdifferenz 1940-1960 und 2013-2023 (°C)', values=df_parallel_num['AverageTemperatureDifference']),
        dict(range=[df_parallel_num['AverageSurfaceTemperature'].min(), df_parallel_num['AverageSurfaceTemperature'].max()],
             label='Durchschnittl. Oberflächentemperatur (°C)', values=df_parallel_num['AverageSurfaceTemperature']),
        dict(range=[df_parallel_num['Total'].min(), df_parallel_num['Total'].max()],
             label='Durchschnittl. Treibhausgasemissionen (1990 bis 2020) (t)', values=df_parallel_num['Total']),
    ]
))

# Update layout
fig_11.update_layout(
    title={'text': 'Vergleich von Durchschnittstemperatur, Temperaturveränderung und Emissionsbeitrag nach Kontinent', 'x':0.5, 'xanchor':'center'},
    xaxis=dict(title='Attributes'),
    yaxis=dict(title='Value')
)



##################################################################################### Format Dash app #####################################################################################

# Initialize Dash app
app = dash.Dash(__name__, update_title=None)

server = app.server

# 2. Plot: Initialize a boolean variable to track animation state
running_animation = False


# 3. Plot: Preprocess data
df_mean_1940_1959 = df_temperature_by_country_owd.loc[(df_temperature_by_country_owd["Year"] >= 1940) & (df_temperature_by_country_owd["Year"] < 1960), ["Year", "Country", "AverageSurfaceTemperature"]].groupby("Country")["AverageSurfaceTemperature"].mean().to_frame().rename(columns={"AverageSurfaceTemperature": "AverageTemperature_40_59"})
df_interactive = df_temperature_by_country_owd.merge(df_mean_1940_1959, on="Country", how="left")
df_interactive["AverageTemperatureDifference"] = df_interactive["AverageSurfaceTemperature"] - df_interactive["AverageTemperature_40_59"]


# 4. Plot: Preprocess data
df_interactive_g5 = df_emissions_by_country_owd.rename(columns={"Annual CO₂ emissions (per capita)": "Annual CO₂ emissions"})

# Filter out elements which are not countries
unwanted_regions_g5 = ['Asia', 'Asia (excl. China and India)', 'Europe', 'Europe (excl. EU-27)', 'Europe (excl. EU-28)', 'European Union (27)', 'European Union (28)', 'South America', 'World', 'North America', 'North America (excl. USA)', 'Oceania', 'High-income countries', 'Low-income countries', 'Lower-middle-income countries', 'Upper-middle-income countries']
filtered_countries_g5 = [country for country in df_interactive_g5['Country'].unique() if country not in unwanted_regions_g5]

# Define options for the dropdown menu with subgroups
dropdown_options_g5 = [
    {'label': 'Nach Region', 'options': [
        {'label': region, 'value': region} for region in ['Asia', 'Asia (excl. China and India)', 'Europe', 'Europe (excl. EU-27)', 'Europe (excl. EU-28)', 'European Union (27)', 'European Union (28)', 'South America', 'World', 'North America', 'North America (excl. USA)', 'Oceania']
    ]},
    {'label': 'Nach Einkommen', 'options': [
        {'label': region, 'value': region} for region in ['High-income countries', 'Low-income countries', 'Lower-middle-income countries',  'Upper-middle-income countries']
    ]},
    {'label': 'Nach Land', 'options': [
        {'label': region, 'value': region} for region in filtered_countries_g5
    ]},
]


# 5. Plot: Preprocess data
df_interactive_g6 = df_total_emissions_by_sectors_owd

# Filter out elements which are not countries
unwanted_regions_g6 = ['Asia', 'Europe', 'European Union (27)', 'South America', 'World', 'North America', 'Oceania', 'High-income countries', 'Low-income countries', 'Lower-middle-income countries', 'Upper-middle-income countries']
filtered_countries_g6 = [country for country in df_interactive_g6['Country'].unique() if country not in unwanted_regions_g6]

# Define options for the dropdown menu with subgroups
dropdown_options_g6 = [
    {'label': 'Nach Region', 'options': [
        {'label': region, 'value': region} for region in ['Asia', 'Europe', 'European Union (27)', 'South America', 'World', 'North America', 'Oceania']
    ]},
    {'label': 'Nach Einkommen', 'options': [
        {'label': region, 'value': region} for region in ['High-income countries', 'Low-income countries', 'Lower-middle-income countries',  'Upper-middle-income countries']
    ]},
    {'label': 'Nach Land', 'options': [
        {'label': region, 'value': region} for region in filtered_countries_g6
    ]},
]

# Define options for the emission dropdown menu
emission_dropdown_options = [
    {'label': 'Alle Treibhausgase', 'value': 'Combined'},
    {'label': 'Kohlenstoffdioxid (CO₂)', 'value': 'CO2'},
    {'label': 'Methan (CH₄)', 'value': 'Methane'},
    {'label': 'Distickstoffmonoxid (N₂O)', 'value': 'Nitrious oxide'}]


# 6. Plot: External Plotly graph objects
fig_7_ext = go.Figure(fig_7)
fig_8_ext = go.Figure(fig_8)
fig_9_ext = go.Figure(fig_9)
fig_10_ext = go.Figure(fig_10)

# Site switch variable
site_switch = 0

# Content sets
content_sets = [
    {
        "info": "Die Artenvielfalt, auch Biodiversität genannt, ist das reichhaltige Zusammenspiel von Tier- und Pflanzenarten sowie ihrer Lebensräume auf der Erde. Der Living Planet Index ist ein wichtiger Indikator für die Gesundheit und Vielfalt der globalen Tierwelt. Er zeigt die Veränderungen der populationsbasierten Tierbestände über die Zeit hinweg und bietet Einblicke in den Zustand der Artenvielfalt auf unserem Planeten. In den letzten Jahrzehnten hat der Living Planet Index besorgniserregende Trends aufgezeigt, die auf eine zunehmende Bedrohung der Artenvielfalt hindeuten. Einige der Hauptursachen für den Rückgang der Tierbestände sind der Verlust und die Zerstörung von Lebensräumen, die Übernutzung von Ressourcen, der Klimawandel, Umweltverschmutzung, Wilderei und invasive Arten.",
        "picture": "https://imgs.mongabay.com/wp-content/uploads/sites/30/2022/07/18202321/Effects-of-pollution-fishdeath2.jpg",
        "graph": fig_7_ext,
        "header": "1. Bedrohung der Artenvielfalt"
    },
    {
        "info": "In den letzten Jahrzehnten ist eine besorgniserregende Zunahme von Naturkatastrophen zu beobachten. Die Häufigkeit von klimabedingten Naturkatastrophen hat sich in den letzten Jahrzehnten fast verzehnfacht. Von vernichtenden Stürmen bis hin zu verheerenden Überschwemmungen und Waldbränden - die Liste der häufig auftretenden Ereignisse ist lang. Während Naturkatastrophen durch endogene oder tektonische Ursachen (Vulkanausbrüche, Erdbeben) eher konstant geblieben sind, haben sich die Katastrophen, die durch das Klima bedingt werden, vervielfacht. Extremwetterereignisse wie Stürme und Dürren treten zunehmend häufiger auf. Die Konsequenzen für Mensch und Natur sind verheerend und bringen unzählige Opfer mit sich: Zerstörung von Lebensgrundlagen, ökologische Störungen und immense wirtschaftliche Schäden. Auch die Gletscherschmelze als Folge des Klimawandels hat weitreichende Folgen für die umliegenden Ökosysteme. Neben der Erhöhung des Meeresspiegels können instabile Gletscherseen durch das Brechen von Dämmen katastrophale Überschwemmungen verursachen.",
        "picture": "https://i.guim.co.uk/img/media/e2ab3d757977f12121cbe3ca150f4a7ddb1bfe9d/0_115_5472_3283/master/5472.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=ed9510bc89d67ccd736378f2ae0da8e4",
        "graph": fig_8_ext,
        "header": "2. Naturkatastrophen"
    },
    {
        "info": "Grönland, einst bekannt für seine eisigen Weiten, erlebt eine bemerkenswerte Veränderung: Der Klimawandel hat das Land zu einem unerwarteten Ziel für Landwirtschaft gemacht. In jüngster Zeit wurden auf Grönlands ehemals vereisten Flächen Kartoffeln angebaut, ein Zeichen für die drastischen Veränderungen in der Region. Dies steht im starken Kontrast zu den Schmelzraten der Eismassen in Grönland und der Antarktis, die alarmierend hoch sind. Das Schmelzen der Eismassen in diesen Regionen trägt signifikant zum globalen Meeresspiegelanstieg bei und birgt ernsthafte Folgen für Küstengemeinden weltweit. Es ist eine alarmierende Erinnerung daran, dass der Klimawandel bereits jetzt spürbare Folgen hat. Wissenschaftler beobachten mit Sorge, wie die Eismassen in Grönland und der Antarktis weiter schwinden und betonen die Dringlichkeit, Maßnahmen zum Schutz der Polarregionen und zur Eindämmung des Klimawandels zu ergreifen.",
        "picture": "https://i0.wp.com/www.yesmagazine.org/wp-content/uploads/2019/12/lydon-alaska-glacier-calving.jpg?fit=1920%2C1080&quality=90&ssl=1",
        "graph": fig_9_ext,
        "header": "3. Schmelzen der Eisflächen"
    },
    {
        "info": "Amsterdam und Venedig, zwei bekannte Beispiele für Städte, die mit dem steigenden Meeresspiegel konfrontiert sind, stehen vor einer ständigen Herausforderung: Das Schützen gegen die zunehmende Flut. In den Niederlanden, einem Land, das größtenteils unter dem Meeresspiegel liegt, ist der Anstieg des Meeresspiegels eine unmittelbare und ernsthafte Bedrohung. Doch die Herausforderungen, die die Niederlande bewältigen müssen, sind kein Einzelfall - sie spiegeln eine globale Realität wider. Diese Situation verdeutlicht das globale Problem des Meeresspiegelanstiegs, der durch das Abschmelzen der Eismassen in Grönland und der Antarktis sowie die thermische Ausdehnung der Ozeane verursacht wird. Diese Phänomene haben bereits dazu geführt, dass Küstenlinien weltweit zurückgedrängt wurden und ganze Inselstaaten gefährdet sind. Der Anstieg des Meeresspiegels stellt eine zunehmende Herausforderung für die weltweite Infrastruktur, Wirtschaft und Sicherheit dar. Es ist daher von entscheidender Bedeutung, dass wir dringende Maßnahmen ergreifen, um den Klimawandel einzudämmen und uns an die bereits irreversiblen Auswirkungen des Meeresspiegelanstiegs anpassen.",
        "picture": "https://i.natgeofe.com/n/ac1a33f0-bcac-4405-8cfc-76f737e4e894/02-Rising-Seas-Cities_3x4.jpg",
        "graph": fig_10,
        "header": "4. Meeresspiegelanstieg"
    }
]


# Plot 7: Define graph
fig_11_ext = go.Figure(fig_11)



# Define custom styles
styles = {
    'dropdown': {'font-family': 'Arial, sans-serif'},
    'graph': {'font-family': 'Arial, sans-serif'}
}

# Define app layout
app.title = "Klima im Wandel"
app.layout = html.Div(
    style={
        'width': '80%',
        'margin': 'auto',
        'border-radius': '20px',
        'padding': '0px',
        'background-color': 'white',
    },
    children=[
        # Header 1
        html.H1('Klima im Wandel', style={'font-family': 'Arial, sans-serif', 'text-align': 'center', 'margin-bottom': '50px'}),

        # Subtitle 1
        html.H2('Eine Analyse der globalen Erwärmung, der weltweiten Treibhausgas-Emissionen und der damit einhergehenden Folgen', style={'font-family': 'Arial, sans-serif', 'text-align': 'center', 'margin-bottom': '10px'}),
        html.P('Sidney Krause', style={'font-family': 'Arial, sans-serif', 'text-align': 'center', 'margin-bottom': '50px'}),
        html.Div([
            html.Img(
                src='https://ideas.ted.com/wp-content/uploads/sites/3/2022/04/FINAL_Climate-reading-list.jpg',
                style={'width': '100%', 'height': '100%', 'object-fit': 'cover', 'object-position': '50% 20%'}
            ),
        ], style={'width': '100%', 'height': '400px', 'margin-bottom': '50px', }),

        html.H2('1. Globale Erwärmung und Klimaschutz', style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown('''Die globale Erwärmung ist eine der größten Herausforderungen unserer Zeit: Die ansteigende Konzentration von Treibhausgasen in der Atmosphäre, hauptsächlich verursacht durch menschliche Aktivitäten wie die Verbrennung fossiler Brennstoffe und die Entwaldung, führen zu einem Anstieg der globalen Durchschnittstemperatur. Bereits heute hinterlässt dies gravierende Spuren auf unserer Erde:
                Vom Abschmelzen von Gletschern und Eisdecken, den Anstieg des Meeresspiegels, häufigere und intensivere Extremwetterereignisse wie Stürme über Dürren und Überschwemmungen bis hin zu Veränderungen in Ökosystemen und der Verbreitung von Krankheiten sind die dokumentierten Auswirkungen drastisch.
                Dabei sind die Folgen des Klimawandels nicht nur auf Umweltbereiche beschränkt, sondern haben auch weitreichende soziale und wirtschaftliche Konsequenzen. 
                Die Bezeichnung „Klimawandel“ wird in der Alltagssprache oftmals mit „globaler Erwärmung“ gleichgesetzt.  Dies trifft allerdings nicht den Kern des Begriffs: Tatsächlich bezeichnet der Klimawandel sowohl die die Erwärmung als auch die Abkühlung des globalen Klimas über eine lange Zeitraumsbetrachtung.<sup>1</sup><br><br>
                In einer Welt, die von wandelnden Landschaften, bedrohten Arten und extremen
                Wetterereignissen geprägt ist, wird die Dringlichkeit des Klimaschutzes klar wie nie zuvor.
                Eine stetig anwachsende Weltbevölkerung sowie auf Wachstum basierender
                Wohlstand sehen sich einer begrenzten Ressourcenverfügbarkeit gegenübergestellt.
                Während Klimaveränderungen oft weltweite Auswirkungen haben, gibt es aber auch
                Bereiche, in denen die Folgen zunächst im regionalen Rahmen bleiben. Aufgrund der
                hochkomplexen Wechselwirkungen lassen sich Ursache-Wirkungs-Zusammenhänge oft nicht
                sicher ausmachen. Zu den größten Herausforderungen des 21. Jahrhunderts zählen nicht nur den Schutz von Eisbergen und Regenwäldern, sondern auch der Erhalt unserer natürlichen Lebensgrundlage. Letztendlich ist Klimaschutz nicht nur wichtig – er ist unverzichtbar für das Überleben unseres Planeten und seiner Bewohner. Jede Veränderung, die wir heute vornehmen, ist ein Keimling der Hoffnung für morgen.<br><br>
                Die im Folgenden visualisierten Daten sollen helfen, die Ursachen und Folgen global steigender Temperaturen in interaktiver Weise aufzuzeigen. Der Artikel zielt darauf ab, eine Übersicht über das Thema Klimawandel zu erhalten, indem auf ausgewählte Zusammenhänge eingegangen wird.''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),

        html.H2('2. Temperaturveränderung mit Zukunftsszenarien', style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown('''Die untenstehende Grafik veranschaulicht, wie sich die durchschnittliche Oberflächentemperatur auf der Erde von 1850 bis 2015 verändert hat. Dabei werden die Jahresdurchschnitte aller Länder gemittelt. Das Konfidenzintervall, die durchschnittliche Messunsicherheit, nimmt mit zunehmender Annäherung an die Gegenwart ab. Dies lässt sich auf eine höhere Anzahl an Messungen und präzisere Instrumente zurückführen. Insgesamt lässt sich ein eindeutiger Aufwärtstrend ausmachen: Die globale Durchschnittstemperatur steigt seit der vorindustriellen Zeit signifikant an. Die globale Erwärmung wird in der Grafik eindeutig dargestellt.<br><br>
                Wenn wir über Temperaturveränderungen sprechen, öffnen sich allerdings auch Türen zu verschiedenen Zukunftsszenarien, die auf wissenschaftlichen Modellen und Statistiken basieren. Hierfür gibt es optimistische und pessimistische Methoden:
                Optimistische Prognosen rechnen mögliche Wege, wie wir durch gemeinsame Anstrengungen und innovative Technologien den Temperaturanstieg begrenzen können, mit ein. Pessimistische Prognosen hingegen gehen von stärker ansteigenden Emissionen aus.<br><br>
                In untenstehender Grafik wird eine äußerst simple und reduzierte Zeitreihendatenprognose mithilfe von Polynomregression angewendet. Diese soll lediglich die Grundannahmen veranschaulichen. Geht man davon aus, dass der aktuelle globale Treibhausgas-Ausstoß auf demselben Niveau verharrt, so stellt die schwarze Linie die Temperaturprognose bei sonst gleichbleibenden Bedingungen bis zum Jahr 2100 dar. Der pessimistische Fall, hier die rote Linie, geht von zunehmenden Treibhausgas-Emissionen in der Zukunft aus. Der grüne Graph stellt ein optimistisches Szenario, nämlich eine Abnahme globaler Emissionen, dar. Dabei wird von einer Abnahme um denselben Wert wie bei der Zunahme im pessimistischen Szenario ausgegangen. Auch wenn es sich hier nur um ein simples Modell handelt, so wird dennoch die Bedeutung der globalen Emissionen auf unsere künftige Durchschnittstemperatur klar. <br><br>
                Welches Szenario letztlich Realität wird, hängt von diversen Faktoren ab, nicht zuletzt auch überregionalen politischen Maßnahmen, dem technologischen Fortschritt und dem Verhalten der Gesellschaft. Eine konkrete Maßnahme stellt dabei das Pariser Klimaschutzabkommen von 2016 dar. Dieses wegweisende Abkommen zwischen 197 Staaten setzt sich das Ziel, den Anstieg der globalen Durchschnittstemperatur bis zum Jahr 2100 deutlich unter 2 °C über dem vorindustriellen Niveau zu halten und Anstrengungen zu unternehmen, um die Erwärmung auf 1,5 °C Celsius zu begrenzen.<sup>2</sup> Der vorindustrielle Zeitraum ist dabei nicht klar definiert. Meistens wird für das vorindustrielle Niveau daher der Durchschnitt der Oberflächentemperatur von 1850 bis 1900 angesetzt.<sup>3</sup> Letzteres Ziel wird durch die graue Linie dargestellt. Die prognostizierten Werte zeigen jedoch, dass die langfristige Erreichung der Klimaziele äußerst fraglich ist. 
                ''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # Dashboard 1: temperature development
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.H2('Historischer Temperaturverlauf mit möglichen Zukunftsszenarien', style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center', 'text-align': 'center'}),
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Radio button to toggle between modes
                        html.Div([
                            html.Label("Ansichtsmodus:", style={**styles['graph'], 'font-weight': 'bold'}),
                            html.Div([
                                dcc.RadioItems(
                                    id='mode-selector',
                                    options=[
                                        {'label': 'Historischer Temperaturverlauf', 'value': 'g2_mode1'},
                                        {'label': 'Historischer Temperaturverlauf mit möglichen Zukunftsszenarien', 'value': 'g2_mode2'}
                                    ],
                                    value='g2_mode1',
                                    style={'display': 'flex', 'flex-direction': 'column', 'font-family': 'Arial, sans-serif', 'margin-between': '20px'}
                                )
                            ], style={'margin-top': '10px'})
                        ])
                    ]
                ),

                # 1. Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                    },
                    children=[
                        dcc.Graph(
                            id='temperature-graph',
                            figure={
                                'data': [g2_graph_1, g2_graph_2, g2_graph_3, g2_graph_4, g2_graph_5, g2_graph_6, g2_graph_7, g2_graph_8],
                                'layout': {
                                    'xaxis': {'title': 'Jahr', 'tickmode': 'linear', 'tick0': df_temperature_global_kag["Year"].min(), 'dtick': 5, 'tickangle': 90, 'tickfont': {'size': 10}, 'ticks': 'outside', 'showline': True, 'linecolor': 'black', 'gridcolor': None},
                                    'yaxis': {'title': 'Durchschnittliche Bodentemperatur (ºC)', 'ticks': 'outside', 'showline': True, 'linecolor': 'black', 'gridcolor': 'lightgrey'},
                                    'plot_bgcolor': 'white',
                                    'legend': {'x': 0.5, 'y': 1.1, 'font': {'size': 10}}
                                }
                            }
                        )
                    ]
                )
            ]
        ),
        html.H2('3. Temperaturentwicklung einzelner Länder im Überblick', style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown('''Die Temperaturentwicklung einzelner Länder ist ein Schlüsselaspekt bei der Analyse der Auswirkungen des Klimawandels auf regionaler Ebene. In vielen Teilen der Welt sind signifikante Temperaturveränderungen zu beobachten, die sowohl ökologische als auch sozioökonomische Auswirkungen haben. Diese Veränderungen können sich in verschiedenen Regionen der Welt unterschiedlich manifestieren. <br><br>
                Das erste Schaubild dieses Kapitels befasst sich mit der absoluten Veränderung der jährlichen Durchschnittstemperatur im Vergleich zur Referenzperiode von 1940 bis 1959. Das Zurückgreifen auf die jährliche Temperaturdifferenz statt der jährlichen Durchschnittstemperatur ermöglicht eine eindeutigere Darstellung, da die Streuung der Werte hier in einem kleineren Intervall stattfindet. Dadurch zeigen sich die Verfärbungen der Karte deutlicher. Grundsätzlich fällt in den Blick, dass der Temperaturanstieg in der nördlichen Hemisphäre deutlich stärker als auf der Südhalbkugel ausfällt. Nichtsdestotrotz ist bei fast allen Ländern ein Temperaturanstieg ablesbar. Die überwiegende Mehrheit der Temperaturentwicklungen verschiedener Länder deutet auf einen signifikanten Anstieg der globalen Durchschnittstemperaturen hin.<br><br>
                Einige Länder erleben eine rapide Erwärmung, die zu Dürren, Waldbränden und dem Abschmelzen von Gletschern führt, während andere mit verstärkten Niederschlägen und erhöhter Hochwassergefahr konfrontiert sind. 
                Beispielsweise zeigt die hier folgend Visualisierung, dass arktische Länder wie Grönland und Russland eine überdurchschnittliche Erwärmung erfahren, was zu drastischen Veränderungen in der Eisschmelze, im Permafrost und in der Artenvielfalt führt. Dies setzt weitere Treibhausgase frei und endet in einem Teufelskreis aus Emissionen und Erwärmung. <br><br>
                Die Kenntnis über regionale Temperaturentwicklungen ist von großer Bedeutung.
                In tropischen Regionen können steigende Temperaturen zu vermehrten Hitzewellen, Wasserknappheit und Ernteausfällen führen. Länder mit Küstenregionen stehen wiederum vor der Bedrohung durch steigende Meeresspiegel und zunehmende Sturmfluten. Die Temperaturänderungen können sich außerdem negativ auf die landwirtschaftliche Produktion, die Wasserversorgung, die menschliche Gesundheit und die Artenvielfalt auswirken. Die Temperaturentwicklung einzelner Länder ist also ein wichtiger Aspekt des globalen Klimawandels und erfordert daher sowohl nationale als auch internationale Anstrengungen, um die Auswirkungen zu mindern und sich an die veränderten Bedingungen anzupassen. 
                ''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # 2. Dashboard: temperature by country (map)
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.H2('Temperaturveränderung einzelner Länder im Zeitverlauf', style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center', 'text-align': 'center'}),
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex',
                        'justify-content': 'space-between',
                        'align-items': 'center',
                    },
                    children=[
                        # Menu for starting animation
                        html.Div([
                            html.Label("Jahrwahl und Animation:", style=styles['graph']),
                            html.Div([
                                dcc.Slider(
                                    id='year-slider-g2',
                                    min=df_interactive['Year'].min(),
                                    max=df_interactive['Year'].max(),
                                    value=df_interactive['Year'].min(),
                                    marks={str(year): str(year) if year % 5 == 0 else '' for year in df_interactive['Year'].unique()},
                                    step=1,
                                ),
                            ], style={'margin-top': '10px', 'width': '90%'}),
                        ], style={'flex-grow': 1, 'width': '100%'}),
                        html.Div([
                            html.Button('Play', id='play-button', n_clicks=0),
                            html.Button('Pause', id='pause-button', n_clicks=0),
                            dcc.Interval(
                                id='interval-component',
                                interval=1000,
                                n_intervals=0
                            ),
                        ], style={'margin-top': '10px', 'width': '10%'})
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex',
                        'font-family': 'Arial, sans-serif',
                    },
                    children=[
                        # Menu for selecting view
                        html.Div([
                            html.Label("Anischtsmodus:", style=styles['graph']),
                            html.Div([
                                dcc.RadioItems(
                                    id='projection-selector',
                                    options=[
                                        {'label': 'Kartenansicht', 'value': 'natural'},
                                        {'label': 'Globusansicht', 'value': 'globe'}
                                    ],
                                    value='natural',
                                ),
                            ], style={'margin-top': '10px', 'display': 'flex', 'flex-direction': 'column'})
                        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
                    ]
                ),
                # 2. Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'flex-wrap': 'wrap',
                    },
                    children=[
                        html.Div([
                            dcc.Graph(id='choropleth-map'),
                        ])
                    ]
                )
            ]
        ),
        dcc.Markdown('''Basierend auf der visualisierten Temperaturentwicklung stellt sich die Frage, wo der Mensch in den kommenden Jahrzehnten überhaupt noch leben kann. Um dies zu beantworten, muss der durchschnittliche Jahrestemperaturverlauf herangezogen und auf die verschiedenen Regionen aufgeschlüsselt werden. Die jährlichen Jahrestemperaturveränderungen zeigen die dynamische Entwicklung des globalen Klimas im Laufe der Zeit. In den letzten Jahrzehnten hat sich ein deutlicher Trend zu steigenden Temperaturen weltweit abgezeichnet. Rekordjahre mit extremen Hitzeperioden treten häufiger auf, und die durchschnittlichen Temperaturen steigen kontinuierlich an. So war 2023 weltweit das wärmste Jahr seit Beginn der Wetteraufzeichnungen.<sup>4</sup> Auch Länder in Nordamerika und Südeuropa verzeichnen mittlerweile Extremtemperaturen, die lange Zeit nur in der Sahelzone erdenklich waren.<br><br> 
                Mithilfe des folgenden interaktiven Liniendiagramms lässt sich die Temperaturentwicklung mehrerer Länder gleichzeitig darstellen und so miteinander vergleichen. Hierbei kann zwischen der absoluten Temperaturdifferenz gegenüber dem Durchschnitt der Jahresdurchschnittstemperaturen einer selbst wählbaren Referenzperiode oder der Oberflächendurchschnittstemperatur gewählt werden. Ersteres eignet sich gut, um Temperaturentwicklungen vergleichen zu können, Zweiteres, um Durchschnittstemperaturen verschiedener Länder abgleichen zu können. Der Darstellungszeitraum ist im Intervall von 1940 bis 2024 frei wählbar.
                ''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # 3. Dashboard: temperature by country
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.H2('Temperaturveränderung auf Länderbene', style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center', 'text-align': 'center'}),
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Dropdown menu for selecting countries
                        html.Div([
                            html.Label("Länder:", style=styles['graph']),
                            html.Div([
                                dcc.Dropdown(
                                    id='country-dropdown',
                                    options=[{'label': country, 'value': country} for country in df_interactive['Country'].unique()],
                                    value=["Germany"],
                                    multi=True,
                                    style=styles['dropdown'],
                                    placeholder='Bitte wählen Sie Länder aus'
                                ),
                            ], style={'margin-top': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                        # Range slider for selecting displayed time range
                        html.Div([
                            html.Label("Darstellungszeitraum:", style=styles['graph']),
                            html.Div([
                                dcc.RangeSlider(
                                    id='year-slider',
                                    min=df_interactive['Year'].min(),
                                    max=df_interactive['Year'].max(),
                                    value=[1970, 2024],
                                    marks={str(year): str(year) for year in range(df_interactive['Year'].min(), df_interactive['Year'].max() + 1, 10)},
                                    step=1,
                                    included=True,
                                    allowCross=False
                                ),
                            ], style={'display': 'inline-block', 'width': '90%', 'margin-top': '10px'}),
                            html.Div([
                                html.Button("\U000021BA", id="reset-year-slider", n_clicks=0, style={'font-size': '20px'}),
                            ], style={'display': 'inline-block', 'width': '5%', 'vertical-align': 'top'}),
                        ], style={'width': '48%', 'float': 'right', 'margin-left': '4%', 'margin-right': '0', 'vertical-align': 'top', 'display': 'inline-block', 'justify-content': 'center'}),
                    ]
                ),
                # Add notification if too many countries are selected
                html.Div(id='notification', style={'margin-top': '10px','margin-bottom': '10px', 'margin-left': '5px'}),

                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Dropdown menu for switching between temperature types
                        html.Div([
                            html.Label("Temperaturdarstellung:", style=styles['graph']),
                            html.Div([
                                dcc.Dropdown(
                                    id='temperature-type-dropdown',
                                    options=[
                                        {'label': 'Temperaturdifferenz zum Durchschnitt der Referenzperiode', 'value': 'AverageTemperatureDifference'},
                                        {'label': 'Durchschnittstemperatur', 'value': 'AverageTemperature'}
                                    ],
                                    value='AverageTemperatureDifference',
                                    style=styles['dropdown']
                                ),
                            ], style={'margin-top': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                        # Range slider for selecting reference time range
                        html.Div([
                            html.Label("Referenzperiode:", style=styles['graph']),
                            html.Div([
                                dcc.RangeSlider(
                                    id='custom-mean-slider',
                                    min=df_interactive['Year'].min(),
                                    max=df_interactive['Year'].max(),
                                    value=[1940, 1960],
                                    marks={str(year): str(year) for year in range(df_interactive['Year'].min(), df_interactive['Year'].max() + 1, 10)},
                                    step=1,
                                    included=True,
                                    allowCross=False,
                                ),
                            ], style={'display': 'inline-block', 'width': '90%', 'margin-top': '10px'}),
                            html.Div([
                                html.Button("\U000021BA", id="reset-custom-mean-slider", n_clicks=0, style={'font-size': '20px'}),
                            ], style={'display': 'inline-block', 'width': '5%', 'vertical-align': 'top'}),
                        ], id='custom-mean-container', style={'width': '48%', 'float': 'right', 'margin-left': '4%', 'margin-right': '0', 'vertical-align': 'top', 'display': 'inline-block', 'justify-content': 'center'}),
                    ]
                ),
                # 3. Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                    },
                    children=[
                        dcc.Graph(id='line-chart', style=styles['graph'])
                    ]
                )
            ]
        ),
        html.H2('4. Treibhausgas-Emissionen je Land',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}),
        html.Div([
            html.Img(
                src='https://static.euronews.com/articles/stories/07/83/14/10/1200x675_cmsv2_b69ff6c5-ff9b-5443-a46d-5a381dc20df0-7831410.jpg',
                style={'width': '100%', 'height': '100%', 'object-fit': 'cover', 'object-position': '30% 20%'}
            ),
        ], style={'width': '100%', 'height': '400px', 'margin-bottom': '50px', }),
        dcc.Markdown('''Die CO<sub>2</sub>-Emissionen pro Land oder Region sind ein wichtiger Indikator für den Beitrag eines Landes zum globalen Klimawandel. Die Pro-Kopf-Emissionen berücksichtigt darüber hinaus die Bevölkerungsgröße und ermöglichen einen Vergleich zwischen Ländern mit unterschiedlicher Einwohnerzahl. Einige Länder sind dabei für einen überproportional großen Anteil der weltweiten emittierten Treibhausgase, insbesondere Kohlenstoffdioxid verantwortlich. So spiegeln sich die Innovativität, wirtschaftliche Lage, Zusammensetzung der Energiesektoren, Bevölkerungsdichte sowie externe Faktoren im dargestellten Diagramm wider.<br><br>
                Katar ist dabei das Land mit den höchsten Pro-Kopf-Emissionen weltweit aufgrund seiner starken Abhängigkeit von Erdöl und Erdgas, die den Großteil seiner Wirtschaft ausmachen. Die Vereinigten Staaten haben historisch gesehen ebenfalls hohe Emissionen pro Kopf verursacht, obwohl ihre Emissionen in den letzten Jahren aufgrund verstärkter Bemühungen zur Dekarbonisierung rückläufig waren. Auffallend sind zudem die erhöhten Pro-Kopf-Emissionen Ozeaniens aufgrund seiner starken Abhängigkeit von Kohle für die Stromerzeugung und seiner ausgedehnten Viehzuchtindustrie. Es gibt jedoch auch Länder und Regionen mit niedrigen Pro-Kopf-Emissionen, wie z.B. viele europäische Länder mit einem hohen Anteil erneuerbarer Energien und einer effizienten Ressourcenallokation. Dabei soll angemerkt sein, dass Länder mit niedrigeren Pro-Kopf-Emissionen nicht unbedingt als vorbildlich betrachtet werden sollten (zum Beispiel China). Schließlich können ihre Gesamtemissionen aufgrund ihrer großen Bevölkerung dennoch erheblich sein. Betrachtet man durch Verändern des dargestellten Zeitraums die Entwicklungen im postindustriellen Zeitalter, so formen die Kurven heutiger Industriestaaten ein umgedrehtes „U“. Zunächst steigen die Emissionen infolge des wachsenden Wohlstandes an und fallen nach Erreichen eines Höhepunkts wieder ab. Dieser Rückgang kann eingeleiteten Umweltschutzmaßnahmen zugeschrieben werden. Mit Blick auf Asien zeigt sich: Je später ein Land dem Prozess der Industrialisierung beitritt, desto schneller wird das Maximum an Pro-Kopf-Emissionen erreicht. Dieses Szenario könnte in aufstrebenden Schwellenländern wie China in den folgenden Dekaden eintreten. 
                ''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # 4. Dashboard: emissions by country
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.H2('C0₂-Emissionen nach Regionen und Ländern', style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center', 'text-align': 'center'}),
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Dropdown menu for selecting countries
                        html.Div([
                            html.Label("Länder und Regionen:", style=styles['graph']),
                            html.Div([
                                dcc.Dropdown(
                                    id='country-dropdown-g5',
                                    options=sum([
                                        [{'label': group['label'], 'value': group['label'], 'disabled': True}] + group['options']
                                        for group in dropdown_options_g5
                                    ], []),
                                    value=['Asia', 'Europe', 'Africa', 'Oceania', 'South America', 'North America'],
                                    multi=True,
                                    style=styles['dropdown'],
                                    placeholder="Wählen Sie Länder oder Regionen aus",
                                ),
                            ], style={'margin-top': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                        # Range slider for selecting displayed time range
                        html.Div([
                            html.Label("Darstellungszeitraum:", style=styles['graph']),
                            html.Div([
                                dcc.RangeSlider(
                                    id='year-slider-g5',
                                    min=df_interactive_g5['Year'].min(),
                                    max=df_interactive_g5['Year'].max(),
                                    value=[1800, 2022],
                                    marks={str(year_g5): str(year_g5) for year_g5 in range(df_interactive_g5['Year'].min(), df_interactive_g5['Year'].max() + 1, 20)},
                                    step=1,
                                    included=True,
                                    allowCross=False
                                ),
                            ], style={'display': 'inline-block', 'width': '90%', 'margin-top': '10px'}),
                            html.Div([
                                html.Button("\U000021BA", id='reset-year-slider-g5', n_clicks=0, style={'font-size': '20px'}),
                            ], style={'display': 'inline-block', 'width': '5%', 'vertical-align': 'top'}),
                        ], style={'width': '48%', 'float': 'right', 'margin-left': '4%', 'margin-right': '0', 'vertical-align': 'top', 'display': 'inline-block', 'justify-content': 'center'}),
                    ]
                ),
                # Add notification if too many countries are selected
                html.Div(id='notification-g5', style={'margin-top': '10px','margin-bottom': '10px', 'margin-left': '5px'}),

                # 4. Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                    },
                    children=[
                        dcc.Graph(id='line-chart-g5', style=styles['graph'])
                    ]
                )
            ]
        ),
        html.H2('5. Treibhausgas-Emissionen nach Sektoren',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown('''Ohne den natürlichen Treibhauseffekt würde die durchschnittliche Temperatur auf der Erde nicht bei den aktuellen 15 °C liegen, sondern bei frostigen -18 ° C. Grundsätzlich haben Treibhausgase in der Atmosphäre also nicht ausschließlich negative Auswirkungen. In den letzten Jahrhunderten sind jedoch zusätzliche, durch Menschen verursachte Emissionen hinzugekommen, die ein besorgniserregendes Ausmaß erreicht haben. Die Folge davon ist die zuvor grafisch dargestellte Erderwärmung. Zu den am meisten klimaschädlichen Gasen gehören Kohlendioxid (CO<sub>2</sub>), Methan (CH<sub>4</sub>) und Distickstoffmonoxid (N<sub>2</sub>O/Lachgas).<br><br>
                Die Auswirkungen der jeweiligen Emissionsarten können im folgenden Diagramm mithilfe der Filterfunktionalitäten einzeln veranschaulicht werden. Die Emissionen nach Sektor geben Aufschluss darüber, aus welchen Quellen Treibhausgasemissionen hauptsächlich stammen und ermöglichen eine gezielte Analyse von Emissionsquellen und deren mögliche Reduzierung. Die Hauptsektoren, die für Treibhausgasemissionen verantwortlich sind umfassen Energieerzeugung, Transport, Industrie, Landwirtschaft, Verkehr und Gebäude.<br><br>
                In vielen Industrieländern sind die Emissionen aus dem Energiesektor der größte Beitrag zum Gesamtausstoß. Länder wie China und die Vereinigten Staaten haben einen hohen Anteil an Emissionen aus der Industrie aufgrund ihrer umfangreichen Produktion und Nutzung von Kohle, Öl und Gas. Der Verkehrssektor trägt ebenfalls erheblich zur Emission bei, insbesondere in Ländern mit einer hohen Motorisierungsrate wie die USA oder Deutschland.
                In Entwicklungsländern sind die Emissionen aus der Landwirtschaft oft bedeutend, insbesondere durch Viehzucht und landwirtschaftliche Praktiken wie Reisanbau. Dies ist zum Beispiel in Ländern wie Brasilien und Indien der Fall. Die Entwaldung ist ebenfalls eine bedeutende Quelle von Treibhausgasemissionen in einigen Entwicklungsländern, besonders in Ländern mit großen Regenwaldgebieten wie Brasilien und Indonesien.<br><br>
                In Bezug auf die Unterschiede je nach Einkommenshöhe der Länder lassen sich bestimmte Trends erkennen. In wohlhabenderen Ländern sind die Emissionen aus dem Energiesektor oft erhöht, da sie sich auf energieintensive Industrien wie Produktion und Technologie konzentrieren. In ärmeren Ländern hingegen können die Emissionen aus der Landwirtschaft und Entwaldung dominieren, da sie eine größere Rolle in der Wirtschaft spielen. Gleichzeitig zeigt sich, dass Schwellenländer allmählich zu Industrienationen avancieren, was sich in wachsenden Emissionswerten widerspiegelt. Länder mit hohem Pro-Kopf-Einkommen hingegen haben mehr finanzielle Mittel für klimafreundliche Innovationen zur Verfügung.<br><br>
                Um den Klimawandel wirksam zu bekämpfen, ist es entscheidend, die Hauptquellen von Treibhausgasemissionen zu identifizieren und entsprechende Gegenmaßnahmen zu ergreifen. Dies erfordert eine differenzierte, also eine sektor- und länderspezifische Herangehensweise, welche die spezifischen Herausforderungen und Potenziale jedes einzelnen Landes individuell berücksichtigt.
                ''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # 5. Dashboard: emissions by sector
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.H2('Treibhausgas-Emissionen nach Sektoren', style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center', 'text-align': 'center'}),
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Dropdown menu for selecting emission type
                        html.Div([
                            html.Label("Emissionsart:", style=styles['graph']),
                            html.Div([
                                dcc.Dropdown(
                                    id='emission-dropdown-g6',
                                    options=emission_dropdown_options,
                                    value="Combined",
                                    multi=False,
                                    style=styles['dropdown'],
                                    placeholder='Bitte wählen Sie die Emissionsart aus'
                                ),
                            ], style={'margin-top': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    ]
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Dropdown menu for selecting countries
                        html.Div([
                            html.Label("Länder und Regionen:", style=styles['graph']),
                            html.Div([
                                dcc.Dropdown(
                                    id='country-dropdown-g6',
                                    options=sum([
                                        [{'label': group['label'], 'value': group['label'], 'disabled': True}] + group['options']
                                        for group in dropdown_options_g6
                                    ], []),
                                    value='World',
                                    multi=False,
                                    style=styles['dropdown'],
                                    placeholder="Wählen Sie ein Land oder eine Region aus",
                                ),
                            ], style={'margin-top': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                        # Range slider for selecting displayed time range
                        html.Div([
                            html.Label("Darstellungszeitraum:", style=styles['graph']),
                            html.Div([
                                dcc.RangeSlider(
                                    id='year-slider-g6',
                                    min=df_interactive_g6['Year'].min(),
                                    max=df_interactive_g6['Year'].max(),
                                    value=[1990, 2020],
                                    marks={str(year_g6): str(year_g6) for year_g6 in range(df_interactive_g6['Year'].min(), df_interactive_g6['Year'].max() + 1, 5)},
                                    step=1,
                                    included=True,
                                    allowCross=False
                                ),
                            ], style={'display': 'inline-block', 'width': '90%', 'margin-top': '10px'}),
                            html.Div([
                                html.Button("\U000021BA", id='reset-year-slider-g6', n_clicks=0, style={'font-size': '20px'}),
                            ], style={'display': 'inline-block', 'width': '5%', 'vertical-align': 'top'}),
                        ], style={'width': '48%', 'float': 'right', 'margin-left': '4%', 'margin-right': '0', 'vertical-align': 'top', 'display': 'inline-block', 'justify-content': 'center'}),
                    ]
                ),

                # 5. Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                    },
                    children=[
                        dcc.Graph(id='area-chart-g6', style=styles['graph'])
                    ]
                )
            ]
        ),
        html.H2('6. Auswirkungen der globalen Erwärmung',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown('Mithilfe der beiden Knöpfe in der Kopfzeile kann man zwischen verschiedenen Auswirkungen navigieren.',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # Dashboard: Effects of climate change
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.Button('‹', id='previous-button-g7', n_clicks=0,
                                    style={'font-size': '30px', 'margin-right': '20px'}),
                        html.H2('', id='header-g7',
                                style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center',
                                       'text-align': 'center'}),
                        html.Button('›', id='next-button-g7', n_clicks=0, style={'font-size': '30px'})
                    ],
                ),
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '0px',
                        'margin-bottom': '15px',
                        'display': 'flex'
                    },
                    children=[
                        # Information
                        html.Div([
                            html.P('', id='info-container-g7', style=styles['graph']),
                        ], style={'width': '48%', 'margin-right': '2%', 'border-radius': '20px', 'padding': '10px',
                                  'background-color': 'white', 'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                                  'display': 'block', 'vertical-align': 'top', 'justify-content': 'center'}),

                        # Picture
                        html.Div([
                            html.Div(id='picture-container-g7',
                                     style={'width': '100%', 'height': '100%', 'border-radius': '20px',
                                            'background-size': 'cover', 'background-position': 'center'}),
                        ], style={'width': '48%', 'margin': '0 auto', 'padding': '0px', 'background-color': 'white',
                                  'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)', 'display': 'flex',
                                  'justify-content': 'center'}),
                    ]
                ),

                # Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                    },
                    children=[
                        dcc.Graph(id='graph-container-g7', style=styles['graph'])
                    ]
                )
            ]
        ),
        html.H2('7. Wie steht es um Klimagerechtigkeit?',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown('''Die allgemeinen Zusammenhänge zwischen Temperaturveränderung, Durchschnittstemperatur und Emissionsbeitrag sind eng miteinander verbunden und treiben den Klimawandel voran. Die folgende Grafik bietet eine Gesamtübersicht der bereits betrachteten Faktoren und ermöglicht, ein überregionales Ursache-Wirkungs-Muster zu visualisieren. Dabei stellen die einzelnen Linien je ein Land dar. Die Gruppierung findet nach Kontinenten statt. Auch das Thema Klimagerechtigkeit lässt sich in der Grafik ablesen: In welchem Ausmaß spürt ein Land die Temperatursteigerung im Vergleich zu den Emissionen, die es ausschüttet?<br><br>
                Der Emissionsbeitrag jedes Kontinents variiert je nach wirtschaftlicher Entwicklung, Bevölkerungsdichte, Energiesystemen und Landnutzung. Länder mit einem hohen Pro-Kopf-Emissionsniveau wie die USA und Kanada tragen einen erheblichen Anteil zum globalen Emissionsausstoß bei, während aufstrebende Volkswirtschaften wie China und Indien aufgrund ihrer Bevölkerungsgröße und industriellen Aktivitäten ebenfalls bedeutende Emittenten sind. Entwicklungsländer hingegen haben oft niedrigere Emissionsbeiträge, leiden jedoch häufig stärker unter den Auswirkungen des Klimawandels aufgrund ihrer geringeren Anpassungskapazitäten und hohen Vulnerabilität. Dies ist zum Beispiel bei vielen afrikanischen Ländern der Fall. <br><br>
                Der Zusammenhang zwischen Durchschnittstemperatur, Temperaturveränderung und Emissionsbeitrag variiert auf jedem Kontinent und prägt die individuelle Situation im Angesicht des Klimawandels. Was sich definitiv zeigt: Klimagerechtigkeit gibt es leider nicht.
                ''',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),


        # 7. Dashboard: parallel coordinates
        html.Div(
            style={
                'margin': 'auto',
                'border-radius': '20px',
                'padding': '15px',
                'background-color': '	#E8E8E8',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                'margin-bottom': '50px',
            },
            children=[
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '5px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                        'margin-bottom': '15px',
                        'font-family': 'Arial, sans-serif',
                        'align-items': 'center',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center'
                    },
                    children=[
                        html.H2('Zusammenhang von Durchschnittstemperatur, Temperaturveränderung und Emissionsbeitrag nach Kontinent', style={'display': 'inline-block', 'flex': '1', 'justify-content': 'center', 'text-align': 'center'}),
                    ]
                ),

                # 7. Graph container
                html.Div(
                    style={
                        'margin': 'auto',
                        'border-radius': '20px',
                        'padding': '20px',
                        'background-color': 'white',
                        'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.1)',
                    },
                    children=[
                        dcc.Graph(figure={
                            'data': fig_11_ext.data,
                            'layout': fig_11_ext.layout}, id='parallel_coordinates-chart', style=styles['graph'])
                    ]
                )
            ]
        ),
        dcc.Store(id='current-index', data=0),
        html.H2('8. Fazit',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        dcc.Markdown(
            '''Zusammenfassend lässt sich feststellen, dass die vorliegenden Daten ein alarmierendes Bild zeichnen. Die Emissionen von Treibhausgasen, insbesondere CO2, steigen weiterhin kontinuierlich an, hauptsächlich aufgrund menschlicher Aktivitäten wie der Verbrennung fossiler Brennstoffe und dem Transport. Diese Emissionen haben zu einem spürbaren Anstieg der globalen Durchschnittstemperaturen geführt, der sich in einer Vielzahl von physischen und ökologischen Veränderungen manifestiert.<br><br>Die Auswirkungen des Klimawandels sind bereits heute deutlich sichtbar. Von zunehmenden Extremwetterereignissen über das Abschmelzen von Gletschern und Polareis bis hin zum Anstieg des Meeresspiegels - die Folgen des Klimawandels sind vielfältig und weitreichend. In diesem Artikel konnte selbstverständlich nur ein kleiner Teil möglicher Folgen beleuchtet werden.<br><br>Angesichts dieser Herausforderungen ist es von entscheidender Bedeutung, dass wir dringende Maßnahmen ergreifen, um die Emissionen von Treibhausgasen zu reduzieren und die Anpassungsfähigkeit unserer Gesellschaften und Ökosysteme zu stärken. Wir müssen zusammenarbeiten, um die Ziele des Pariser Abkommens zu erreichen und den Anstieg der globalen Durchschnittstemperatur auf unter 2 °C zu begrenzen. Letztendlich liegt es an uns allen - Regierungen, Unternehmen, Gemeinschaften und Einzelpersonen -, den Klimawandel als eine existenzielle Bedrohung anzuerkennen und gemeinsam konkrete Maßnahmen zu ergreifen, um eine lebenswerte Zukunft für alle zu sichern. Indem wir jetzt handeln, können wir die Weichen für eine nachhaltige und widerstandsfähige Zukunft stellen, in der Mensch und Natur im Einklang leben können. Es ist nie zu spät für Klimaschutz!''',
            style={'font-family': 'Arial, sans-serif', 'margin-bottom': '50px'}, dangerously_allow_html=True),

        html.H2('9. Quellen',
                style={'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}),
        html.P('Datensätze:', style={'font-family': 'Arial, sans-serif', 'font-weight': 'bold'}),
        html.P(
            'https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview [aufgerufen am: 24.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data/data [aufgerufen am: 25.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'Global Carbon Budget (2023); Population based on various sources (2023) – with major processing by Our World in Data. “Annual CO₂ emissions (per capita) – GCB” [dataset]. Global Carbon Project, “Global Carbon Budget”; Various sources, “Population” [original data]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://www.climatewatchdata.org/data-explorer/historical-emissions?historical-emissions-data-sources=climate-watch&historical-emissions-gases=all-ghg&historical-emissions-regions=All%20Selected&historical-emissions-sectors=total-including-lucf%2Ctotal-including-lucf&page=1 [aufgerufen am: 27.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P('https://stats.livingplanetindex.org/ [aufgerufen am: 27.04.2024]',
               style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://ourworldindata.org/grapher/natural-disasters-by-type#sources-and-processing [aufgerufen am: 27.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P('https://ourworldindata.org/grapher/ice-sheet-mass-balance [aufgerufen am: 27.04.2024]',
               style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level) [aufgerufen am: 27.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P('Text:', style={'font-family': 'Arial, sans-serif', 'font-weight': 'bold'}),
        html.P('(1) https://www.wwf.de/themen-projekte/klimaschutz/klimawandel [aufgerufen am: 30.04.2024]',
               style={'font-family': 'Arial, sans-serif'}),
        html.P(
            '(2) https://www.bmwk.de/Redaktion/DE/Artikel/Industrie/klimaschutz-abkommen-von-paris.html) [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            '(3) https://www.mdr.de/nachrichten/deutschland/panorama/klimaziel-erderwaermung-vergleichswert-100.html#:~:text=Der%20Referenzwert%20des%20vorindustriellen%20Zeitraumes,es%20etwa%207%2C5%20Grad. [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            '(4) https://www.tagesschau.de/wissen/klima/wetter-jahresbilanz-dwd-100.html [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'Nutzung von generativer künstlicher Intelligenz (ChatGPT, OpenAI) für Formulierungshilfen und Rechtschreibkorrektur: https://chat.openai.com',
            style={'font-family': 'Arial, sans-serif'}),
        html.P('Bildadressen:', style={'font-family': 'Arial, sans-serif', 'font-weight': 'bold'}),
        html.P(
            'https://ideas.ted.com/wp-content/uploads/sites/3/2022/04/FINAL_Climate-reading-list.jpg  [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://static.euronews.com/articles/stories/07/83/14/10/1200x675_cmsv2_b69ff6c5-ff9b-5443-a46d-5a381dc20df0-7831410.jpg [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://imgs.mongabay.com/wp-content/uploads/sites/30/2022/07/18202321/Effects-of-pollution-fishdeath2.jpg [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://i.guim.co.uk/img/media/e2ab3d757977f12121cbe3ca150f4a7ddb1bfe9d/0_115_5472_3283/master/5472.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=ed9510bc89d67ccd736378f2ae0da8e4 [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://i0.wp.com/www.yesmagazine.org/wp-content/uploads/2019/12/lydon-alaska-glacier-calving.jpg?fit=1920%2C1080&quality=90&ssl=1 [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.P(
            'https://i.natgeofe.com/n/ac1a33f0-bcac-4405-8cfc-76f737e4e894/02-Rising-Seas-Cities_3x4.jpg [aufgerufen am: 30.04.2024]',
            style={'font-family': 'Arial, sans-serif'}),
        html.Br(),
        html.P('© Sidney Krause, 2024', style={'font-family': 'Arial, sans-serif'}),
    ]
)



##################################################################################### Add functions and callbacks #####################################################################################

# 3. Plot: Function to get discrete colors from a sequential colorscale
def get_discrete_colors_from_sequential(colorscale, num_colors):
    return px.colors.sequential.YlOrRd_r[0:num_colors]

# 4. Plot: Function to get discrete colors from a sequential colorscale
def get_discrete_colors_from_sequential_g5(colorscale, num_colors):
    return px.colors.sequential.Oranges_r[0:num_colors]

# 5. Plot: Function to get discrete colors from a sequential colorscale
def get_discrete_colors_from_sequential_g6(colorscale, num_colors):
    return px.colors.sequential.Viridis[0:num_colors]



# 1. Callback to update graph based on mode selection
@app.callback(
    dash.dependencies.Output('temperature-graph', 'figure'),
    [dash.dependencies.Input('mode-selector', 'value')]
)
def update_graph(mode):
    if mode == 'g2_mode1':
        data = [g2_graph_1, g2_graph_2, g2_graph_3]
    else:
        data = [g2_graph_1, g2_graph_2, g2_graph_3, g2_graph_4, g2_graph_5, g2_graph_6, g2_graph_7, g2_graph_8]

    return {
        'data': data,
        'layout': {
            'xaxis': {'title': 'Jahr', 'tickmode': 'linear', 'tick0': df_temperature_global_kag["Year"].min(), 'dtick': 10, 'tickangle': 90, 'tickfont': {'size': 10}, 'ticks': 'outside', 'showline': True, 'linecolor': 'black', 'gridcolor': None},
            'yaxis': {'title': 'Durchschnittliche Bodentemperatur (ºC)', 'ticks': 'outside', 'showline': True, 'linecolor': 'black', 'gridcolor': 'lightgrey'},
            'plot_bgcolor': 'white',
            'legend': {'font': {'size': 10}}
        }
    }


# 2.Callback to update choropleth map based on slider value
@app.callback(
    [Output('choropleth-map', 'figure'),
     Output('year-slider-g2', 'value')],
    [Input('interval-component', 'n_intervals'),
     Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('projection-selector', 'value'),
     Input('choropleth-map', 'relayoutData')],
    [State('year-slider-g2', 'value')]
)
def update_map(n_intervals, play_button, pause_button, projection_type, relayout_data, selected_year):
    global running_animation

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None

    if triggered_id == 'play-button.n_clicks':
        running_animation = True

    if triggered_id == 'pause-button.n_clicks':
        running_animation = False

    if running_animation:
        # Calculate the next year based on the current slider value
        selected_year += 1
        if selected_year > df_interactive['Year'].max():
            selected_year = df_interactive['Year'].min()

    filtered_df = df_interactive[df_interactive['Year'] == selected_year]

    fig = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color="AverageTemperatureDifference",
        color_continuous_scale="OrRd",
        range_color=(df_interactive["AverageTemperatureDifference"].min(), df_interactive["AverageTemperatureDifference"].max()),
        hover_name="Country",
        hover_data={"AverageTemperatureDifference": ":.2f"},
        labels={"AverageTemperatureDifference": "Temperaturveränderung ggü. 1940-1959 (°C)", "Year": "Jahr"},
        title="Durchschnittliche Jahrestemperatur im Vergleich<br>zum Durchschnitt der Jahre 1940 bis 1960",
        template="plotly_white",
        custom_data=["Year"]
    )

    if projection_type == 'globe':
        fig.update_geos(projection_type="orthographic")
    else:
        fig.update_geos(projection_type="natural earth")

    # Add hovertemplate
    fig.update_traces(hovertemplate="<b>%{location}</b><br>Jahr: %{customdata[0]}<br>Temperaturveränderung ggü. 1940-1959: %{z:.2f}°C<extra></extra>")

    # Customize layout
    fig.update_layout(
        coloraxis_colorbar=dict(title='Temperaturveränderung (°C)', x=1.05),
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Arial"),
        dragmode='orbit',
        title_x=0.5,
        title_y=0.9,
        title_xanchor='center',
        title_yanchor='middle',
    )

    # Check if relayout_data contains 'dragmode' key and update running_animation based on that
    if relayout_data and 'dragmode' in relayout_data:
        if relayout_data['dragmode'] == 'orbit':
            running_animation = False

    return fig, selected_year


# 3. Callback
@app.callback(
    [Output('line-chart', 'figure'),
     Output('notification', 'children'),
     Output('year-slider', 'value'),
     Output('custom-mean-slider', 'value'),
     Output('custom-mean-container', 'style')],
    [Input('country-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('temperature-type-dropdown', 'value'),
     Input('custom-mean-slider', 'value'),
     Input('reset-year-slider', 'n_clicks'),
     Input('reset-custom-mean-slider', 'n_clicks')],
    [State('year-slider', 'min'),
     State('year-slider', 'max'),
     State('custom-mean-slider', 'min'),
     State('custom-mean-slider', 'max')]
)
def update_line_chart(selected_countries, selected_years, selected_temp_type, custom_mean_years, reset_year_clicks, reset_custom_mean_clicks, year_min, year_max, custom_mean_min, custom_mean_max):
    # Reset sliders if reset button is clicked
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'reset-year-slider':
            selected_years = [1970, 2024]
        elif prop_id == 'reset-custom-mean-slider':
            custom_mean_years = [1940, 1960]

    start_year, end_year = selected_years
    custom_mean_start_year, custom_mean_end_year = custom_mean_years

    filtered_df = df_interactive[(df_interactive['Country'].isin(selected_countries)) & (df_interactive['Year'] >= start_year) & (df_interactive['Year'] <= end_year)]

    # If selected temperature type is the difference from the custom mean
    if selected_temp_type == 'AverageTemperatureDifference':
        # Limit number of selected countries to a maximum of 6
        if len(selected_countries) > 6:
            notification_message = html.Div([
                html.Span('Bitte wählen Sie maximal sechs Länder aus!', style=styles['dropdown'])
            ])
            return dash.no_update, notification_message, selected_years, custom_mean_years, {'width': '48%', 'float': 'right', 'margin-left': '4%', 'margin-right': '0', 'vertical-align': 'top', 'display': 'inline-block', 'justify-content': 'center'}
        else:
            custom_mean_df = df_interactive[(df_interactive['Year'] >= custom_mean_start_year) & (df_interactive['Year'] < custom_mean_end_year)]
            custom_mean_values = custom_mean_df.groupby('Country')['AverageSurfaceTemperature'].mean().reset_index()
            filtered_df = filtered_df.merge(custom_mean_values, on='Country', how='left', suffixes=('', '_custom_mean'))
            filtered_df['CustomTemperatureDifference'] = filtered_df['AverageSurfaceTemperature'] - filtered_df['AverageSurfaceTemperature_custom_mean']

            traces = []
            num_countries = len(selected_countries)
            colors = get_discrete_colors_from_sequential(px.colors.sequential.YlOrRd, num_countries)
            for i, country in enumerate(selected_countries):
                filtered_country_df = filtered_df[filtered_df['Country'] == country]
                trace = go.Scatter(x=filtered_country_df['Year'], y=filtered_country_df['CustomTemperatureDifference'], mode='lines',
                                   name=f'{country}', line=dict(color=colors[i]), hovertemplate="<b>" + country + '</b><br>Temperaturveränderung ggü. ' + str(custom_mean_start_year) + ' - ' + str(custom_mean_end_year) + ': %{y:.2f}°C<extra></extra>')
                traces.append(trace)
            layout = go.Layout(title={'text':'Absolute Veränderung der Jahresdurchschnittstemperatur <br> im Vergleich zum Durchschnitt der Referenzperiode ' + str(custom_mean_start_year) + ' - ' + str(custom_mean_end_year), 'x':0.5, 'xanchor':'center'},
                               xaxis=dict(title="Jahr", tickmode="linear", tick0=start_year, dtick=5, tickangle=0, tickfont=dict(size=12), ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey", zeroline=True, zerolinecolor="lightgrey", zerolinewidth=1),
                               yaxis=dict(title="Temperaturveränderung (ºC)", ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey", zeroline=True, zerolinecolor="lightgrey", zerolinewidth=1),
                               plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)')
            fig = go.Figure(data=traces, layout=layout)

            return fig, None, selected_years, custom_mean_years, {'width': '48%', 'float': 'right', 'margin-left': '4%', 'margin-right': '0', 'vertical-align': 'top', 'display': 'inline-block', 'justify-content': 'center'}

    # If the selected temperature type is the average surface temperature
    else:
        # Limit the number of selected countries to a maximum of 6
        if len(selected_countries) > 6:
            notification_message = html.Div([
                html.Span('Bitte wählen Sie maximal sechs Länder aus!', style=styles['dropdown'])
            ])
            return dash.no_update, notification_message, selected_years, custom_mean_years, {'display': 'none'}
        else:
            traces = []
            num_countries = len(selected_countries)
            colors = get_discrete_colors_from_sequential(px.colors.sequential.YlOrRd, num_countries)

            for i, country in enumerate(selected_countries):
                filtered_country_df = filtered_df[filtered_df['Country'] == country]
                trace = go.Scatter(x=filtered_country_df['Year'], y=filtered_country_df['AverageSurfaceTemperature'], mode='lines',
                                   name=f'{country}', line=dict(color=colors[i]), hovertemplate="<b>" + country + "</b><br>Durchschnittliche Oberflächentemperatur im Jahr %{x}: %{y:.2f}°C<extra></extra>")
                traces.append(trace)
            layout = go.Layout(title={'text':'Jahresdurchschnittstemperatur im zeitlichen Verlauf', 'x':0.5, 'xanchor':'center'},
                               xaxis=dict(title="Jahr", tickmode="linear", tick0=str(start_year), dtick=5, tickangle=0, tickfont=dict(size=12), ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey", zeroline=True, zerolinecolor="lightgrey", zerolinewidth=1, range=[int(start_year), int(end_year)]),
                               yaxis=dict(title="Oberflächentemperatur (ºC)", ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey", zeroline=True, zerolinecolor="lightgrey", zerolinewidth=1),
                               plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)')
            fig = go.Figure(data=traces, layout=layout)

            return fig, None, selected_years, custom_mean_years, {'display': 'none'}


# 4. Callback
@app.callback(
    [Output('line-chart-g5', 'figure'),
     Output('notification-g5', 'children'),
     Output('year-slider-g5', 'value')],
    [Input('country-dropdown-g5', 'value'),
     Input('year-slider-g5', 'value'),
     Input('reset-year-slider-g5', 'n_clicks'),],
    [State('year-slider-g5', 'min'),
     State('year-slider-g5', 'max')]
)
def update_line_chart_g5(selected_countries, selected_years, reset_year_clicks, year_min, year_max):
    # Reset sliders if reset button is clicked
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'reset-year-slider-g5':
            selected_years = [1800, 2022]

    start_year_g5, end_year_g5 = selected_years
    filtered_df_g5 = df_interactive_g5[(df_interactive_g5['Country'].isin(selected_countries)) & (df_interactive_g5['Year'] >= start_year_g5) & (df_interactive_g5['Year'] <= end_year_g5)]

    # Limit number of selected countries to a maximum of 6
    if len(selected_countries) > 6:
        notification_message_g5 = html.Div([
            html.Span('Bitte wählen Sie maximal sechs Länder aus!', style=styles['dropdown'])
        ])
        return dash.no_update, notification_message_g5, selected_years
    else:
        traces_g5 = []
        num_countries_g5 = len(selected_countries)
        colors_g5 = get_discrete_colors_from_sequential_g5(px.colors.sequential.Oranges, num_countries_g5)

        for i_g5, country_g5 in enumerate(selected_countries):
            filtered_emission_df_g5 = filtered_df_g5[filtered_df_g5['Country'] == country_g5]
            trace_g5 = go.Scatter(x=filtered_emission_df_g5['Year'], y=filtered_emission_df_g5['Annual CO₂ emissions'], mode='lines',
                                  name=f'{country_g5}', line=dict(color=colors_g5[i_g5]), hovertemplate="<b>" + country_g5 + "</b><br>C0₂-Emissionen je Einwohner im Jahr %{x}: %{y:.2f} Tonnen<extra></extra>")
            traces_g5.append(trace_g5)
        layout_g5 = go.Layout(title={'text': 'C0₂-Emissionen je Einwohner in Tonnen von ' + str(start_year_g5) + ' bis ' + str(end_year_g5), 'x':0.5, 'xanchor':'center'},
                              xaxis=dict(title="Jahr", tickmode="linear", tick0=start_year_g5, dtick=10, tickangle=0, tickfont=dict(size=12), ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey", zeroline=True, zerolinecolor="lightgrey", zerolinewidth=1, range=[int(start_year_g5), int(end_year_g5)]),
                              yaxis=dict(title="C0₂-Emissionen je Einwohner (t)", ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey", zeroline=True, zerolinecolor="lightgrey", zerolinewidth=1),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
        fig_g5 = go.Figure(data=traces_g5, layout=layout_g5)

        return fig_g5, None, selected_years


# 5. Callback function to update the chart based on selected emission
@app.callback(
    [Output('area-chart-g6', 'figure'),
     Output('year-slider-g6', 'value')],
    [Input('emission-dropdown-g6', 'value'),
     Input('country-dropdown-g6', 'value'),
     Input('year-slider-g6', 'value'),
     Input('reset-year-slider-g6', 'n_clicks')],
    [State('year-slider-g6', 'min'),
     State('year-slider-g6', 'max')]
)
def update_emission_chart(selected_emission, selected_country, selected_years, reset_year_clicks, year_min, year_max):
    # Reset sliders if reset button is clicked
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'reset-year-slider-g6':
            selected_years = [1990, 2020]

    start_year_g6, end_year_g6 = selected_years

    # Get the dataframe based on the selected emission type and set variables based on that
    if selected_emission == 'CO2':
        df_interactive_g6 = df_co2_emissions_by_country_owd
        # List important column names of dataframe
        columns_g6 = ['Gebäude', 'Industrie',
                      'Flächennutzung und Forstwirtschaft', 'Verbrennung anderer Brennstoffe',
                      'Transport', 'Herstellung und Bau', 'Diffuse Emissionen',
                      'Elektrizität und Wärme']
        emission_unit = 'CO₂-Emissionen (t)'

    elif selected_emission == 'Methane':
        df_interactive_g6 = df_methane_emissions_by_country_owd
        # List important column names of dataframe
        columns_g6 = ['Landwirtschaft', 'Diffuse Emissionen', 'Abfälle',
                      'Flächennutzung und Forstwirtschaft', 'Verbrennung anderer Brennstoffe',
                      'Industrie']
        emission_unit = 'Methan-Emissionen (t)'

    elif selected_emission == 'Combined':
        df_interactive_g6 = df_total_emissions_by_sectors_owd
        # List important column names of dataframe
        columns_g6 = ['Verbrennung anderer Brennstoffe', 'Schweröl',
                      'Abfälle', 'Gebäude', 'Industrie', 'Diffuse Emissionen',
                      'Landwirtschaft', 'Herstellung und Bau', 'Transport',
                      'Elektrizität und Wärme']
        emission_unit = 'Treibhausgas-Emissionen (t)'

    else:
        df_interactive_g6 = df_nitrous_oxide_emissions_by_country_owd
        # List important column names of dataframe
        columns_g6 = ['Landwirtschaft', 'Industrie',
                      'Verbrennung anderer Brennstoffe', 'Abfälle',
                      'Flächennutzung und Forstwirtschaft', 'Energieerzeugung']
        emission_unit = 'N₂O-Emissionen (t)'


    filtered_df_g6 = df_interactive_g6[(df_interactive_g6['Country']==selected_country) & (df_interactive_g6['Year'] >= start_year_g6) & (df_interactive_g6['Year'] <= end_year_g6)]

    colors_g6 = get_discrete_colors_from_sequential_g6(px.colors.sequential.Viridis, len(columns_g6))

    traces_g6 = []

    for column_g6, color_g6 in zip(columns_g6, colors_g6):
        # Initialize a list to store hover labels for each data point
        hover_labels = []
        # Construct hover label for each data point
        for _, row in filtered_df_g6.iterrows():
            if selected_emission == 'Combined' or selected_emission == 'CO2':
                hover_label = f"<b>{selected_country}</b><br>Jahr: {row['Year']}<br>Emissionen durch {column_g6}: {(row[column_g6] / 1e9):.2f} Mrd. Tonnen<br>Gesamtemissionen: {(row['Total'] / 1e9):.2f} Mrd. Tonnen<br>Anteil am Gesamtausstoß: {(row[column_g6] / row['Total'] * 100):.2f}%<extra></extra>"
                hover_labels.append(hover_label)
            else:
                hover_label = f"<b>{selected_country}</b><br>Jahr: {row['Year']}<br>Emissionen durch {column_g6}: {(row[column_g6] / 1e6):.2f} Mio. Tonnen<br>Gesamtemissionen: {(row['Total'] / 1e6):.2f} Mio. Tonnen<br>Anteil am Gesamtausstoß: {(row[column_g6] / row['Total'] * 100):.2f}%<extra></extra>"
                hover_labels.append(hover_label)


        trace_g6 = go.Scatter(
            x=filtered_df_g6['Year'],
            y=filtered_df_g6[column_g6],
            mode='lines',
            fill='tonexty',
            name=column_g6,
            line=dict(color=color_g6),
            hovertemplate=hover_labels,
            stackgroup='one_g6'
        )
        traces_g6.append(trace_g6)

    layout_g6 = go.Layout(
        title={'text': f'Treibhausgas-Emissionen in Tonnen von {start_year_g6} bis {end_year_g6}', 'x':0.5, 'xanchor':'center'},

        xaxis=dict(
            title="Jahr",
            tickmode="linear",
            tick0=start_year_g6,
            dtick=5,
            tickangle=0,
            tickfont=dict(size=12),
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            zerolinewidth=1,
            range=[int(start_year_g6), int(end_year_g6)]
        ),
        yaxis=dict(
            title=f"{emission_unit}",
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            zerolinewidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig_g6 = go.Figure(data=traces_g6, layout=layout_g6)

    return fig_g6, selected_years


# Callback to update content on button clicks
@app.callback(
    Output("info-container-g7", "children"),
    Output("picture-container-g7", "style"),
    Output("graph-container-g7", "figure"),
    Output("header-g7", "children"),
    Output("current-index", "data"),
    Input("previous-button-g7", "n_clicks"),
    Input("next-button-g7", "n_clicks"),
    State("current-index", "data")
)
def update_content(prev_clicks, next_clicks, current_index):

    # Initialize current_index to 0 if not set
    current_index = current_index or 0

    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    # If triggered by previous button
    if triggered_id == "previous-button-g7" and current_index == 0:
        current_index = 3
    elif triggered_id == "previous-button-g7" and current_index != 0:
        current_index -= 1
    # If triggered by next button
    elif triggered_id == "next-button-g7" and current_index == 3:
        current_index = 0
    elif triggered_id == "next-button-g7" and current_index != 3:
        current_index += 1

    content = content_sets[current_index]

    info_content = html.P(content["info"], style=styles['graph'])
    picture_style = {
        'width': '100%',
        'height': '100%',
        'background-image': f'url("{content["picture"]}")',
        'background-size': 'cover',
        'background-position': 'center'
    }
    graph_content = {
        'data': content["graph"].data,
        'layout': content["graph"].layout
    }
    header_content = html.P(content["header"], style={**styles['graph'], 'margin-top': '0px', 'margin-bottom': '0px'})

    return info_content, picture_style, graph_content, header_content, current_index



##################################################################################### Run app #####################################################################################

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)