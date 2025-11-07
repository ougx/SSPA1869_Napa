

# Global Settings

The golbal settings are used to set up the running environment and default values for some of the functions. 

- [*include*]: include other configration file.
- [*masterdir*]: specify the the working directory.
- [*epsg*]: specify the default Coordinate Reference System.
- [*start_date*] the default starting time for time related data.
- [*time_unit*] the default time unit for time related data, default value is `'D'`.
- [*style*] or [*plotstyle*] the matplotlib plotting style, see [style reference](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) for available styles. default value is `'ggplot'`.
- [*err_as_oms*] when `err_as_oms = true`, `residual = observed - simulated`; otherwise `residual = simulated - observed`; default is `false`.
<!-- [*time_unit*] default time unit e.g. 'day', 'minute' -->
<!-- na_value = -999 -->

# Operations

Operations are the commands in `cHydroP` 
## prepare

## extract

The `extract` command will read the MODFLOW head file or MT3D concentration file at the observation wells. It supports horizontal interpolation and different vertical avervging methods (see ).

- mf: 
- [alllayer]: whether to extract heads from all layers; when it is `true`, it will overwrite the `avgMethod` column in the well CSV.

## stat

## pdf

# Data Objects

## mobject2d

The **`mobject2d`** is one of the top data class used in the code to represent 2D data. It cannot be used in the configuration file. But it provides some universal parameters used by its subclasses:

- *rename*: rename the columns of the dataframe of the data object; see [rename](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html).
- *filter*: filter the dataframe of the data object; its expression see [query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).
- [*exclude*]: exclude some data from plotting; however, if the `plotarg_exclude` is specified, the excluded data will also be plotted using `plotarg_exclude`. This is useful, .e.g., when you want to plot the data with zero observation weights.
- [*plotarg*]:         the plotting arguments as a dictionary for the dataset.
- [*plotarg_exclude*]: the plotting arguments as a dictionary for the excluded dataset.
- [*label_exclude*]: the name for the excluded dataset, used in the legend
- [*writedata*]: save the data to a CSV file.
- [*subtract*]:  the dataframe of the current data object is used to subtract the dataframe of another data object.
- [*add*]:       the dataframe of the current data object is used to add the dataframe of another data object.
- [*sub*]:       the dataframe of the current data object is used to subtract the dataframe of another data object.
- [*mul*]:       the dataframe of the current data object is used to multiply the dataframe of another data object.
- [*div*]:       the dataframe of the current data object is used to divide the dataframe of another data object.
- [*calculate*]: the dataframe of the current data object is euqal to the result of the a simple mathmatic expression, e.g. `(simheadA - simHeadB) - (simheadC - simHeadD)` where `simheadA`, `simheadB`, `simheadC` and `simheadD` are `mfbin` objects. The expression will be evaluated by the python `eval` function.

## mobject3d

The **`mobject3d`** is a subclass of **`mobject2d`**. **`mobject3d`** is another top data class used in the code to represent 3D or 4D data and should not be directly used in the configuration file.

## wellprop

The **`wellprop`** is a subclass of **`mobject2d`**. It reprsents a well property table for monitoring wells.

- *source*: file path of the well property file. It can be CSV, XLSX or shpaefile.
- *idcolname*: the well ID field name in the well property table.
- [*plotarg*]: the plotting arguments when plotting the wells on the map; see [matplotlib.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html).
- [*plotarg_loc*]: the plotting arguments when highlighting the well on the map in the hydrogrpha plots; see [matplotlib.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html).

Note: in the well table, it reads the following fields/columns (case insensitive):

- *wellid*: this is the unique name for each well. the name for this column can be specified by `idcolname`.
- *x*: world x coordinate of the well (it should use the same length unit as the MODFLOW model).
- *y*: world y coordinate of the well (it should use the same length unit as the MODFLOW model).
- *niterp*: the number of cells that are used for horizontal interpolation in a layer.
- *avgmethod*: vertical averaging method for wells screening multiple layers. The acceptable values include: 
  - `alllayer`: extract values of all the layers at the well location.
  - `layer`: extract values for a single layer specified in the `layer` field.
  - `thickness`: extract values by thickness-weighted average; this method will require the scrren top and bottom elevations (`scntop` and `scnbot` fields).
  - `trans`: extract values by transmissivity-weighted average; this method will require the scrren top and bottom elevations (`scntop` and `scnbot` fields).
  - `mass`: extract values by mass balance method by calculating a steady state well water level under a fixed pumping rates when  groundwater. This rate needs to be specified in the field `extractq`. The mass balance method is particularly suitable for extracting concentration values. This method will also require the scrren top and bottom elevations (`scntop` and `scnbot` columns).
- [*row1*, *row2*, *row3* .. *row(n)*]: the row numbers for the cells for horizontal interpolation in a layer. the number of these rows should match `niterp`. If there are more rows are specified than `ninterp`, only the first `ninterp` rows will be used. The `prepare` command can be used to calculate these fields using the `bilinear` interpolation.
- [*column1*, *column2*, *column3* .. *column(n)*]: the column numbers for the cells for horizontal interpolation in a layer. the number of these columns should match `niterp`. If there are more columns are specified than `ninterp`, only the first `ninterp` columns will be used. The `prepare` command can be used to calculate these fields using the `bilinear` interpolation.
- [*weight1*, *weight2*, *weight3* .. *weight(n)*]: the interpolation weights (factors) for the cells for horizontal interpolation in a layer. the number of these rows should match `nite
- [*scntop*]: screen top elevation used when layer averging is used or well profile is to plot in the hydrograph plots.
- [*scnbot*]: screen bottom elevation used when layer averging is used or well profile is to plot in the hydrograph plots.
- [*extractq*]: The pumping rate used for groundwater sampling. This value is used when the mass balance method is used to extract values.

## wellobs

The `wellobs` is a subclass of **`mobject2d`**. It reprsents the observation data (e.g. groundwater level or solute concentration) at monitoring wells.

- *source*: file path of the CSV, XLSX or shpaefile.
- *idcolname*: the well ID field name in the data table. The IDs need to match the ones used in the `wellprop` table.
- *valcolname*: the field name of observation values
- [*nmin*]: the minimum number of observation records for a well. If the number of observation records of a well is smaller than this value, the well will be excluded, default is 1.
- [*limits*]: the value bounds for validated values. The observed records with values outside these bounds will be excluded.

Note: in the observation data table, it reads the following fields (not case sensitive):

- *wellid*: this is the unique name for each well. the name for this column can be specified by `idcolname`. The well names will be used to match the ones in the `wellprop` table.
- *time*: the time of the observation in the same unit as the model, referencing the start of the model simulation.
- *value*: the observed values. This field name need to be be specified by `valcolname`.

## raster

**`raster`** is a subclass of **`mobject2d`**. It is used to read and plot a raster file. It can be plot as **`cgrid`** or **`contour`**.

- *source*: a raster file that can be read by [gdal](https://gdal.org/drivers/raster/index.html).

## tseries

**`tseries`** is a subclass of **`mobject2d`**. It is used to read time series of observed or simulated data.
Belows are the options:

- *source*: file path of a the CSV/XLSX file containing the observed and simulated data.
            It must include the `time` column.
            This file can be generated using the `extract` command.
- *idcol*: column name for unit id, e.g. 'wellname'.
- *valcol*: the name[s] of value column[s] of the orginal data; default is all columns.
- [*location*]: a dictionary containing a geospatial data object `shp` or `CSV` and `table` to define the locations of the plotting units,
                e.g. location={shp='huc12.shp'}.
- [*second_y*]: if true, this time series will be plotted using the secondary y-axis.
- [*limits*]: the value bounds for validated values.
- [*nmin*]: minimum number of value entry for each unit.
- [*start_date*]: the time stamp of the starting date; default is the `start_date` defined in the global domain.
- [*time_unit*]: the time unit for the time column; default is the `days` defined in the global domain.
- [*resample*]: a list of the resample rule (calculated as the incremental volume divided by the length of time interval) and the aggregation function, e.g. ['year' 'sum'].
- [*aggfunc*]: the aggregate function used to calculate the aggregation based on the unit id. default is 'sum'.
see [pandas.resample](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling).
- [*plotarg*]: the plotting arguments to plot the time series curve; see [matplotlib.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).


## pstres

**`pstres`** is a subclass of **`tseries`**. It was to read the `.res` file created by `PEST`. The CSV file must include `x` and `y` columns which are used to construct points. **`pstres`** can be added in **`tsplot`**.

- *source*: file path of the `PEST` residual file.
- *wellobs*: the **`wellobs`** name used to match the PEST observations. It must have the colume `obsname` to match the `name` column in the residual file.
- [*label*]: the name used in the legend


## mflst

**`mflst`** is a subclass of **`tseries`**. It is used to read water budget in a MODFLOW list file. 
Belows are the options:

- *source*: file path of a the MODFLOW list file.
- [*volume*]: whether to plot volumes or rates, default is false to plot rates.

## sfrout

**`sfrout`** is a subclass of **`simts`**. It is used to read time series of observations vs simulations.
Belows are the options:

- *source*: file path of a the SFR output file.
- *mf*: name of the MODFLOW object that defines the time discretization for this SFR output.
- [*exttable*]: a dictionary containing one or more `table`, `shp` or `csv` to define auxiliary data,
                must include the ['segment', 'reach'] columns, e.g. exttable={csv='gages'}.

## mfbin

**`mfbin`** is a subclass of **`mobject3d`**. It is used to read the binary MODFLOW or MT3D output.

- *source*: file path of a head or concentration binary file.
- *mf*: the MODFLOW model with which this simulation is generated. this will provide the coordinates of the grid.
- [*minlayer*]: minimum layer for data extraction; data smaller than `minlayer` will be excluded, default is 1.
- [*maxlayer*]: maximum layer for data extraction; data larger than `maxlayer` will be excluded, default is `nlay` of the MF model.
- [*mintime*]: minimum time for data extraction; data earlier than `mintime` will be excluded, default is 0.
- [*maxtime*]: maximum time for data extraction; data later than `maxtime` will be excluded, default is `inf`.
- [*tfunc*]: aggregation function over time dimension, e.g. `tfunc = mean` can calculate the mean groundwater head over time. acceptable functions include `mean`, `min`, `max`, `median`, `sum`.
- [*lfunc*]: aggregation function over layers, e.g. `lfunc = max` can calculate the maximum concentration over the layers. acceptable functions include `mean`, `min`, `max`, `median`, `sum`.
- [*limits*]: the value bounds for validated values. This is useful to exclude dry cells or inactive cells.

## mfarray

**`mfarray`** is a subclass of **`mobject3d`**. It is similar to **`mfbin`** but reads a ascii file containing MODFLOW array data, for example, multi-layer hydraulic conductivity data.

- *source*: the text file containing the data.
- *mf*: the MODFLOW model with which this simulation is generated. this will provide the coordinates of the grid.
- [*dtype*]: data type e.g. `'int'` and `'float'`.
- [*skiprows*]: the number of rows to skip to read the data
- [*nrows*]: the number of rows to read the data (not inlcuding the `skiprows`).
- [*numberwidth*]: the width occupied by each value if the file is fixed-width format.
- [*limits*]: the value bounds for validated values. This is useful to exclude dry cells or inactive cells.
- [*minlayer*]: minimum layer for data extraction; data smaller than `minlayer` will be excluded, default is 1.
- [*maxlayer*]: maximum layer for data extraction; data larger than `maxlayer` will be excluded, default is `nlay` of the MF model.
- [*lfunc*]: aggregation function over layers, e.g. `lfunc = max` can calculate the maximum concentration over the layers. acceptable functions include `mean`, `min`, `max`, `median`, `sum` or a list of these functions.

## mfpkg

**`mfapkg`** is a subclass of **`shp`**. It is used to plot a MODFLOw package.

- *mf*: the MODFLOW model with which this simulation is generated. this will provide the coordinates of the grid.
- *package*: pckage name such as 'riv', 'chd', 'wel', 'drn', 'ghb' and 'lak'.
- *[periods*]: specify what stress period to be used, e.g. `periods = [1, 3]` for stress period 1 and 3, default is all stress periods.
- [*layers*]: specify which layers to be used, e.g. `layers = [1, 3]` for layer 1 and 3, default is all layers.


# Model Objects

## MODFLOW

The **`mf`** object represents a MODFLOW model (support for unsatructed grid in underway).

- *model_ws*: Model workspace path. Default is the current directory or the `masterdir` specified in the global settings.
- *version*: MODFLOW version. Choose one of: "mf2k", "mf2005" (default), "mfnwt", "mfusg" or "mf6".
- [*namefile*]: Path to MODFLOW name file to load. Not needed for `mf6`.
- [*model*]: The submodel in a `mf6` model to be used. If ignored, it will assume to be the first submodel.
- [*xmin*]: world x coordinate of the origin point (lower left corner of model grid).
- [*ymin*]: world y coordinate of the origin point (lower left corner of model grid).
- [*rotation*]: rotation angle of model grid (counter-clockwise), as it is rotated around the origin point.


# Plotting Artists

## shp and csv

**`shp`** is a subclass of **`mobject2d`**. It is used to plot a shapefile in a map. **`csv`** is used to represent points and is a subclass of **`shp`** except that **`csv`** must include `x` and `y` columns which are used to construct points. The parameters for **`shp`** include:

- *source*: the file path of the shapefile (the *.shp file).
- [*filter*]: a statement filter rows, e.g. `'(Model == "SSPA") & (merr <= 0)'` will only include the `Model` column is `SSPA` and the values in the `merr` column are smaller or equal to `0`.
- [*labelpoint*]: the column used to label the symbols when the geometry type is points
- [*zcol*]: the column used to represent the Z coordinates. used for cross section plots
- [*plotarg*]: plotting arguments for the shapefile, see [geopandas.plot](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.plot.html).
- [*legend*]: whether to show legend, default is `false`.
- [*label*]: name used in the legend, default is the name of the object.
- [*size*]: plot points in varying symbol sizes. `size` is the column name used to calculate the size of symbol for the `Point` shapefiles
- [*sizelegend*]: arguments for the symbol size legend when symbols are plotted in varying symbol sizes is. For example, `{loc='lower left', title='Mean errors (feet)'}` defines the location and title of the legend.


## img

**`img`** is a subclass of **`mobject2d`**. It is used to read and plot an image.

- *source*: a image file that can be read by [matplotlib.imread](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html).
- *extent*: placement of the image in the plot as extent (xmin, xmax, ymin, ymax).
- *plotarg*: plotting parameters that can be used for [matplotlib.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html), e.g. `plotarg = {alpha = 0.5, zorder = 0}`.

## cgrid

**`cgrid`** can be used to plot data array as colored grids. **`cgrid`** can be added into **`mapplot`** or **`vplot`**. The data source can be one and only one of the **`mfbin`**, **`mfarray`** or **`raster`**.

- [*mfbin*]: name(s) of one or a list of **`mfbin`** objects.
- [*mfarray*]: name(s) of one or a list of **`mfarray`** objects.
- [*raster*]: name(s) of one or a list of **`raster`** objects.
- [*colorbar*]: a dictionary of arguments for [matplotlib.colorbar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html).
- [*plotarg*]: a dictionary of plotting arguments for the colorgrid, see [matplotlib.pcolormesh](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html).

## contour

**`contour`** is used to represent data as contours. Similar to **`cgrid`**, **`contour`** can be added into **`mapplot`** or **`vplot`**. The data source can be one of the **`mfbin`**, **`mfarray`** or **`raster`**.

- [*mfbin*]: name(s) of one or a list of **`mfbin`** objects.
- [*mfarray*]: name(s) of one or a list of **`mfarray`** objects.
- [*raster*]: name(s) of one or a list of **`raster`** objects.
- [*writeshp*]: export the contour lines to a shpefile.
- [*levels*]: specify the contour levels by a list of values or a integer number for the number of the levels.
- [*dlevel*]: specify the intervel of contour levels. if `levels` are specified then this parameter will be igonred.
- [*limits*]: specify the minimum and maximum values for the contour levels. If this parameter is not specified, the minimum and maximum of the data will be used. if `levels` are specified then this parameter will be igonred.
- [*legend*]: whether to show in the legend, default is false
- [*clabel*]: arguments used to label the contours, see [matplotlib.clabel](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.clabel.html).

## scalebar

**`contour`** can be added to **`mapplot`** to show the scale of map.

- *x*: the x coordinate to place the scalebar.
- *y*: the y coordinate to place the scalebar.
- *length*: the length of the scalebar in scalebar unit.
- [*unit*] the length unit to use for the scalebar, default is 'miles'.
- [*unit_factor*] the unit foctor converting from the map unit to the scalebar unit, dedault is 5280 (feet to miles).
- [*linewidth*] the width of the scalebar, default is 3.
- [*color*] the color of the scalebar, default is 'k'.
- [*pad*] the distance (in points) between the scalebar and the text, default is 5.

## borelog



<!-- TODO -->
## mfarrow

**`mfarrow`** is a subclass of **`mobject3d`**. It reads the CBB file and store the groundwater flux.

- *source*: the MODFLOW list file containing simulated water budget
- *mf*: the MODFLOW model with which this simulation is generated. this will provide the coordinates of the grid.
- [*minlayer*]: minimum layer for data extraction; data smaller than `minlayer` will be excluded, default is 1.
- [*maxlayer*]: maximum layer for data extraction; data larger than `maxlayer` will be excluded, default is `nlay` of the MF model.
- [*mintime*]: minimum time for data extraction; data earlier than `mintime` will be excluded, default is 0.
- [*maxtime*]: maximum time for data extraction; data later than `maxtime` will be excluded, default is `inf`.
- [*tfunc*]: aggregation function over time dimension, e.g. `tfunc = mean` can calculate the mean groundwater head over time. acceptable functions include `mean`, `min`, `max`, `median`, `sum`.
- [*lfunc*]: aggregation function over layers, e.g. `lfunc = max` can calculate the maximum concentration over the layers. acceptable functions include `mean`, `min`, `max`, `median`, `sum`.
- [*limits*]: the value bounds for validated values. This is useful to exclude dry cells or inactive cells.
- [*interval*]: the cell inverval used to place the arrows, it is a two integer list, e.g. [3, 2] means that every 3 rows and 2 columns
- [*plotarg*]: the plotting argument, see [matplotlib.quiver](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html).

# Plot Layouts

## figure 

- [text]: add text to the plot, see [matplotlib.text](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html), e.g. `text = {x=23000, y=45000, s="Model 1", ...}`.



## tsplot

**`tsplot`** is a subcalss of **`figure`** and used to create time series hydrographs. **`scatterplot`** supports multiple data source (which will be horiozntally combined). The data source can be:

- *wellprop*: the name of a **`wellprop`** object. It is used to plot the location maps.
- [*wellobs*]: the name of a **`wellobs`** object to be combined, if defined, observation of **`wellobs`**  will be added to the plot.
- [*pstres*]: the name of a **`pstres`** object to be combined, if defined, simulated values of **`pstres`**  will be added to the plot.
- [*simts*]: one or a list of name(s) of **`simts`** object(s) to be combined, if defined, simulated values of **`simts`**  will be added to the plot.

Other parameters for **`tsplot`** include:  

- [*start_date*] specifies the reference time, default is `start_date` in the master setting.
- [*time_unit*] specifies the time unit, default is `time_unit` in the master setting.
- [*legend_kwds*] is a dictionary of parameter specified the legend, see [matplotlib.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html), e.g. `legend_kwds = {loc=[1.027, 0], facecolor='none'}`.
- [*grouping*]: the column name used to define data catogory, different color will be used between catogories, e.g. `grouping = 'layer'`.
- [*mapplot*]: the name of a **`mapplot`** object to show the well locations.
- [*borelog*]: the name of a **`borelog`** object to show the well logs.



## scatterplot

**`scatterplot`** is a subclass of **`figure`**. It is used to create scatter plots to compare two variables such as observed vs. simulated values or simulated errors (residuals). A linear regression line, calculated based on the data, can be automatically added to the plot. An intervel between 5% and 95% percentiles can also added to the plot. The data source can be:

- [*wellprop*]: the name of a **`wellprop`** object to be combined.
- [*wellobs*]: the name of a **`wellobs`** object to be combined.
- [*pstres*]: the name of a **`pstres`** object to be combined.
- [*simts*]: one or a list of name(s) of **`simts`** object(s) to be combined.

- [*source*]: one or a list of name(s) of file paths to a `csv`, `xlsx` or `shp`. This parameter will be ignored if  any of the other data source is specified. 

Other parameters for **`scatterplot`** include:

- [*xcolname*]: the column name used as the x data, default is 'Observed'.
- [*ycolname*]: one or a list of column name(s) used as the x data, default are the colu,ms after the `xcolname` column.
- [*grouping*]: the column name used to define data catogory, different color will be used between catogories, e.g. `grouping = 'layer'`.
- [*cmap*]: colormap used for different group of points is `grouping` is specified
- [*dotsize*]: the point size for the scatters, default is `10`.
- [*yerror*]: plot the `error` instead of the `y value` when `yerror = true`. 
- [*add_ci*]: whether a shaded area representing intervel between the 5% and 95% percentiles of the error is added to the plot, default is `true`.
- [*add_regression_line*]: whether a regression line is added to the plot, default is `false`.
- [*filter*]: see **`mobject2d`**.
- [*exclude*]: see **`mobject2d`**.

## errcdf

**`errpdf`** is a subcalss of **`scatterplot`** and used to create cumulative probability distribution plot such as simulated errors.
The parameters for **`errpdf`** include:

- [cdfcol]: the column(s) used to calculate the cumumative probability distribution. If not present, the errors are calculated as the differences between the `Observed` column (the 3rd column) and the following columns.
- [stattable_loc]: the location of the error statstics table default is `[0.7, 0.05]`
- [stattable_size]: the width and height of the error statstics table e.g. `[0.24, 0.24]`
- [stattable_fontsize]: the font size for the error statstics table default is `6`

## lstplot

- *mflst*: one or a list of **`mflst`** objects.
- [*volume*]: whether to plot volumes or rates, default is false to plot rates.
- [*unit*]: unit showing in the plot, e.g. `'acrefeet/day'`
- [*unit_factor*] the unit foctor converting from **`mflst`** to the plot unit, dedault is 1.0.
- [*resample*]: resample rule (calculated as the incremental volume divided by the length of time interval), see [pandas.resample](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling).
- [*time_range*]: is a list of two datetime to represent the time range for the plot, default is `None`. 


## mapplot

`mapplot` is used to create maps that can include different types of geospatial data, e.g. Shapefile, Raster, Head/Concentration and CSV. The specified elements will be plotted in the same page (**horizontally concatenated**). The parameters for `mapplot` include:

- extent: map spatial extent expressed as [xmin, xmax, ymin, ymax]
- [epsg]: this is useful when adding spatial data that are not in the same coordinate system as the MODFLOW grid.
- [show_border]: whether to show a black boarder around the map
- [bounds]: relative placement of the map in another plot expressed as [xmin, ymin, width, height]
- [zorder]: zorder relative to other elements in the parent plot
- [basemap]: 'terrain', 'street' or 'satellite'.
- [shp]: name(s) for the shapefile object(s) (it points to the shp block)
- [csv]: name(s) for the csv object(s) (it points to the csv block)
- [img]: name(s) for the image object(s) (it points to the img block)
- [raster]
- [contour]
- [cgrid]
- [mfpkg]


## vplot

`vplot` is used to plot a vertical cross section.

- mf: name of the `mf` object
- [*cshp*]: a horizontal line shp object to define the horizontal location of the cross section. I will use the `row` and `column` fileds in the attribute table or intersection between the geometry and the grid.
- [*showcsline*]: whether plot the cross section line. if not defined, the line is not plotted. the shp must define `zcol` for the z positions.
- [*cpoints*]: a group of x,y pair(s) to define the horizontal location of the cross section; internally it will create a line by connecting the points
- [*cname*]: the cross section name
- [*clim*]: the bounds for the secondary axis
- [*clog*]: whether plotting head/concentration on log scale; default false
- [*clabel*]: label used for the y-axis for head/concentration
- [*plotarg*]: plot arguments for the cross section line
- [*showgrid*]: plot the model grid, default is `true`
- [*contour*]: head/concentration will be plotted as lines on the secondary y axis. the layer specification in the `contour` block will be ignored.
- [*cgrid*]: head/concentration will be plotted as color grid. the layer specification in the `cgrid` block will be ignored.
- [*mfpkg*]: plot the MF package
- [*legend*]: legend arguments or `false` to disable legend
- [*secondy*]: use secondary Y axis to plot the contour data. It can be used to plot concentrations that are not on the same scale as the elevations.
- [*secondy_arg*]: arguments to define secondary Y axis, such as `secondy_arg = {ylim = [1, 1000], yscale='log'}`
