t=["from folium import plugins\n",
"m = folium.Map( location=[10.772095,106.701965], zoom_start=15, tiles='OpenStreetMap')\n",
"# for _, r in data_new.iterrows():\n",
"#     # Without simplifying the representation of each borough,\n",
"#     # the map might not be displayed\n",
"#     sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)\n",
"#     geo_j = sim_geo.to_json()\n",
"#     geo_j = folium.GeoJson(data=geo_j,\n",
"#                            style_function=lambda x: {'fillColor': 'orange'})\n",
"#     folium.Popup(r['Dist_Name']).add_to(geo_j)\n",
"#     geo_j.add_to(m)\n",
"\n",
"#heat_data = [[point.xy[1][0], point.xy[0][0]] for point in points[\"geometry\"] ]\n",
"heat_data = [[point.xy[1][0], point.xy[0][0]] for point in list_points]\n",
"\n",
"heat_data\n",
"plugins.HeatMap(heat_data).add_to(m)\n",
"\n",
"m"]

z = ["delay=5\n",
    "fn=r'./output/heatmap.html'\n",
    "tmpurl='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=fn)\n",
    "m.save(fn)\n",
    "\n",
    "browser = webdriver.Chrome(ChromeDriverManager().install())\n",
    "browser.get(tmpurl)\n",
    "#Give the map tiles some time to load\n",
    "time.sleep(delay)\n",
    "browser.save_screenshot(r'./output/heatmap.png')\n",
    "browser.quit()"]