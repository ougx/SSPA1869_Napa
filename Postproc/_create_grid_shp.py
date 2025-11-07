import flopy
import numpy as np

def create_grid_shp6(shpfile=None, sim_ws=None, modelname=None, xll=None, yll=None, rotation=None, lfac=1.0, crs=None, elev=False):
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, load_only=["dis"], verify_data=False, verbosity_level=0)
    mf = sim.get_model(modelname)
    mf.dis.delr = mf.dis.delr.array * lfac
    mf.dis.delc = mf.dis.delc.array * lfac
    grid = mf.modelgrid
    grid.set_coord_info(xoff=xll, yoff=yll, angrot=rotation, crs=crs)
    if shpfile is None:
        gdf = grid.geo_dataframe
        gdf["row"] = np.repeat(np.arange(1, grid.nrow+1), grid.ncol)
        gdf["column"] = np.tile(np.arange(1, grid.ncol+1), grid.nrow)
        il = 0
        if mf.dis.idomain is not None:
            for ib in mf.dis.idomain.array:
                il += 1
                gdf["ibound" + str(il)] = ib.flat
        if elev:
            gdf["botm0"] = mf.dis.top.array.flatten()
            for il in range(mf.dis.nlay.array):
                gdf[f"botm{il+1}"] = mf.dis.botm.array[il].flatten()
        return gdf
    if elev:
        mf.dis.export(shpfile)
    else:
        mf.dis.idomain.export(shpfile)


shp = create_grid_shp6(sim_ws="../SS", lfac=0.3048, crs="epsg:26913", xll=510063, yll=3587923.64684, elev=True)

active1 = shp[shp.ibound1!=0].dissolve()
active1.to_file("shp_domain1.shp")

active2 = shp[shp.ibound2!=0].dissolve()
active2.to_file("shp_domain2.shp")

active3 = shp[shp.ibound3!=0].dissolve()
active3.to_file("shp_domain3.shp")

shp.to_file("grid.shp")
