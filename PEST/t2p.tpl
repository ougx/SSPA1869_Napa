ptf $
*==============================================================================
* Texture2Par Main Input File
*==============================================================================

BEGIN OPTIONS
  MAX_VSTRUCT         1
  MAX_SECONDARY_DIST  5000
  INFER_LAST_CLASS
  WRITE_NODE_FILES
  #WRITE_DATASET_FILES
  MAX_OUTSIDE_DIST    3000 # feet
END OPTIONS

BEGIN FLOW_MODEL
  # EPSG: 2226, all unit in feet
  TYPE MODFLOW
  NAM_FILE      napa.nam
  TEMPLATE_FILE napa_temp.upw
  XOFFSET       6453400
  YOFFSET       1769800
  ROTATION      20
  HSU_FILE      zones.dat
  PP_ZONE_FILE  zones.dat
END FLOW_MODEL

BEGIN CLASSES
  Coarse
  Fine
END CLASSES

BEGIN DATASET
  FILE     welllog.csv
END DATASET

BEGIN VARIOGRAMS
  # Structure Vtype  Nugget   Sill  Range_min Range_max ang1  nnear
  CLASS Coarse                                              
           1    Sph    0.00   0.12    5000    5000  0.0  20
  CLASS PilotPoints                                         
           1    Sph    0.00   1.0     8000    8000  0.0  6
END VARIOGRAMS

BEGIN PP_LOCS
#ID        X       Y Zone
 1   6383622  1981201  1
 2   6396136  1973964  1
 3   6405324  1967919  1
 4   6415984  1960827  1
 5   6425313  1949593  1
 6   6435537  1938903  1
 7   6446005  1928738  1
 8   6455721  1915243  1
 9   6462472  1906292  1
 10  6470539  1895526  1
 11  6465307  1878157  2
 12  6463856  1873372  2
 13  6384962  1999495  3
 14  6500207  1880218  3
 15  6484398  1891491  4
 16  6490169  1877096  4
 17  6462872  1880463  5
 18  6466933  1870767  5

END PP_LOCS

BEGIN PP_PARAMETERS
  TYPE Global
# ID   KHp          KVp          STp
  1    $khp01   $   $kvp01   $   $stp01   $
  2    $khp02   $   $kvp02   $   $stp02   $
  3    $khp03   $   $kvp03   $   $stp03   $
  4    $khp04   $   $kvp04   $   $stp04   $
  5    $khp05   $   $kvp05   $   $stp05   $
  6    $khp06   $   $kvp06   $   $stp06   $
  7    $khp07   $   $kvp07   $   $stp07   $
  8    $khp08   $   $kvp08   $   $stp08   $
  9    $khp09   $   $kvp09   $   $stp09   $
  10   $khp10   $   $kvp10   $   $stp10   $
  11   $khp11   $   $kvp11   $   $stp11   $
  12   $khp12   $   $kvp12   $   $stp12   $
  13   $khp13   $   $kvp13   $   $stp13   $
  14   $khp14   $   $kvp14   $   $stp14   $
  15   $khp15   $   $kvp15   $   $stp15   $
  16   $khp16   $   $kvp16   $   $stp16   $
  17   $khp17   $   $kvp17   $   $stp17   $
  18   $khp18   $   $kvp18   $   $stp18   $
  TYPE Aquifer
# ID    Class     Kmin         Kmax         Ss           Sy           Aniso        Kd
   1   Coarse     $kminCr01$   $kmaxCr01$   $SSCr01  $   $SyCr01  $   $AnisCr01$   $kdCr01  $
   2   Coarse     $kminCr02$   $kmaxCr02$   $SSCr02  $   $SyCr02  $   $AnisCr02$   $kdCr02  $
   3   Coarse     $kminCr03$   $kmaxCr03$   $SSCr03  $   $SyCr03  $   $AnisCr03$   $kdCr03  $
   4   Coarse     $kminCr04$   $kmaxCr04$   $SSCr04  $   $SyCr04  $   $AnisCr04$   $kdCr04  $
   5   Coarse     $kminCr05$   $kmaxCr05$   $SSCr05  $   $SyCr05  $   $AnisCr05$   $kdCr05  $
   6   Coarse     $kminCr06$   $kmaxCr06$   $SSCr06  $   $SyCr06  $   $AnisCr06$   $kdCr06  $
   7   Coarse     $kminCr07$   $kmaxCr07$   $SSCr07  $   $SyCr07  $   $AnisCr07$   $kdCr07  $
   8   Coarse     $kminCr08$   $kmaxCr08$   $SSCr08  $   $SyCr08  $   $AnisCr08$   $kdCr08  $
   9   Coarse     $kminCr09$   $kmaxCr09$   $SSCr09  $   $SyCr09  $   $AnisCr09$   $kdCr09  $
   10  Coarse     $kminCr10$   $kmaxCr10$   $SSCr10  $   $SyCr10  $   $AnisCr10$   $kdCr10  $
   11  Coarse     $kminCr11$   $kmaxCr11$   $SSCr11  $   $SyCr11  $   $AnisCr11$   $kdCr11  $
   12  Coarse     $kminCr12$   $kmaxCr12$   $SSCr12  $   $SyCr12  $   $AnisCr12$   $kdCr12  $
   13  Coarse     $kminCr13$   $kmaxCr13$   $SSCr13  $   $SyCr13  $   $AnisCr13$   $kdCr13  $
   14  Coarse     $kminCr14$   $kmaxCr14$   $SSCr14  $   $SyCr14  $   $AnisCr14$   $kdCr14  $
   15  Coarse     $kminCr15$   $kmaxCr15$   $SSCr15  $   $SyCr15  $   $AnisCr15$   $kdCr15  $
   16  Coarse     $kminCr16$   $kmaxCr16$   $SSCr16  $   $SyCr16  $   $AnisCr16$   $kdCr16  $
   17  Coarse     $kminCr17$   $kmaxCr17$   $SSCr17  $   $SyCr17  $   $AnisCr17$   $kdCr17  $
   18  Coarse     $kminCr18$   $kmaxCr18$   $SSCr18  $   $SyCr18  $   $AnisCr18$   $kdCr18  $
   1   Fine       $kminFn01$   $kmaxFn01$   $SSFn01  $   $SyFn01  $   $AnisFn01$   $kdFn01  $
   2   Fine       $kminFn02$   $kmaxFn02$   $SSFn02  $   $SyFn02  $   $AnisFn02$   $kdFn02  $
   3   Fine       $kminFn03$   $kmaxFn03$   $SSFn03  $   $SyFn03  $   $AnisFn03$   $kdFn03  $
   4   Fine       $kminFn04$   $kmaxFn04$   $SSFn04  $   $SyFn04  $   $AnisFn04$   $kdFn04  $
   5   Fine       $kminFn05$   $kmaxFn05$   $SSFn05  $   $SyFn05  $   $AnisFn05$   $kdFn05  $
   6   Fine       $kminFn06$   $kmaxFn06$   $SSFn06  $   $SyFn06  $   $AnisFn06$   $kdFn06  $
   7   Fine       $kminFn07$   $kmaxFn07$   $SSFn07  $   $SyFn07  $   $AnisFn07$   $kdFn07  $
   8   Fine       $kminFn08$   $kmaxFn08$   $SSFn08  $   $SyFn08  $   $AnisFn08$   $kdFn08  $
   9   Fine       $kminFn09$   $kmaxFn09$   $SSFn09  $   $SyFn09  $   $AnisFn09$   $kdFn09  $
   10  Fine       $kminFn10$   $kmaxFn10$   $SSFn10  $   $SyFn10  $   $AnisFn10$   $kdFn10  $
   11  Fine       $kminFn11$   $kmaxFn11$   $SSFn11  $   $SyFn11  $   $AnisFn11$   $kdFn11  $
   12  Fine       $kminFn12$   $kmaxFn12$   $SSFn12  $   $SyFn12  $   $AnisFn12$   $kdFn12  $
   13  Fine       $kminFn13$   $kmaxFn13$   $SSFn13  $   $SyFn13  $   $AnisFn13$   $kdFn13  $
   14  Fine       $kminFn14$   $kmaxFn14$   $SSFn14  $   $SyFn14  $   $AnisFn14$   $kdFn14  $
   15  Fine       $kminFn15$   $kmaxFn15$   $SSFn15  $   $SyFn15  $   $AnisFn15$   $kdFn15  $
   16  Fine       $kminFn16$   $kmaxFn16$   $SSFn16  $   $SyFn16  $   $AnisFn16$   $kdFn16  $
   17  Fine       $kminFn17$   $kmaxFn17$   $SSFn17  $   $SyFn17  $   $AnisFn17$   $kdFn17  $
   18  Fine       $kminFn18$   $kmaxFn18$   $SSFn18  $   $SyFn18  $   $AnisFn18$   $kdFn18  $
END PP_PARAMETERS