import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import config
from scipy.stats import pearsonr
from GHGRP_for_Energy.envirofacts_api.Get_GHGRP_data import GHGRP_API
import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw'
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Unit conversions
MT_WEEK_TO_KG_HR = 1000. / (24. * 7.)  # Converts Metric Tons methane per week to kg/hr
MT_QRT_TO_KG_HR =  1000. / (24. * 7. * 13.)  # Converts Metric Tons methane per quarter to kg/hr. 
FT3_PER_SHORT_TON_TO_M3_PER_TON = (0.3048)**3 / 0.907185
MT_QTR_TO_MSCF_DAY = 51921 / 1000 / (7*13)
MSCF_DAY_TO_KG_HR = 1 / MT_QTR_TO_MSCF_DAY * MT_QRT_TO_KG_HR
M3_TO_KG = 0.6802 # https://www.epa.gov/cmop/updated-coal-mine-methane-units-converter @ 15C and 1 atm
KG_TO_MT = 1e-3 # kg to metric tons 
KG_TO_M_T = 1e-3 * 1e-6 # kg to million metric tons 
MST_TO_T = 0.907185 * 1e6 # Million short tons to metric tons 
FT_TO_M = 0.3048
T_TO_M_T = 1e-6 # metric tons to million metric tons
M_T_TO_T = 1e6


def date_to_year_quarter(x):
    return x.dt.year + (x.dt.quarter-1)/4


def correct_typos(table_to_correct, column, typo_replacement_dict):
    """ Corrects typos in the EPA data. """
    for typo, correct in typo_replacement_dict[column].items():
        table_to_correct[column] = table_to_correct[column].str.replace(typo, correct)
    return table_to_correct

def clean_up_msha_id(df, 
                     msha_replace_dict=config.msha_replace_dict, 
                     msha_replace_by_mine_name=config.msha_replace_by_mine_name):
    """ Cleans up MSHA IDs. """
    msha_id_col = df['msha_identification_number'].copy()
    msha_id_col = msha_id_col.str.replace('-','',regex=False).str.replace(' ', '',regex=False).str.replace('&',',',regex=False).str.replace('5.5635239E7','NaN',regex=False)
    msha_id_col = msha_id_col.replace(msha_replace_dict)
    for mine, msha in msha_replace_by_mine_name.items():
        msha_id_col[df['facility_name']==mine] = msha
    return msha_id_col

def correct_num_shafts(df, num_shafts_replacement):
    for repl in num_shafts_replacement:
        print('Fixing number of shafts in: '+repl[0])
        df.loc[(df.facility_name==repl[0])&
                    (df.reporting_year==repl[1])&
                    (df.quarter.str.contains(str(int(repl[2])))), 'num_of_shafts'] = repl[3]
    return df['num_of_shafts']

def get_all_FF_tables(out_dir='../data/FF_tables'):
    """ Retreives all data from the FF section of the EPA GHRP and saves it to a local folder. """
    for table in config.FF_TABLES:
        print(table)
        df = GHGRP_API().get_data(table)
        df.columns = [col.lower() for col in df.columns] # make all columns lowercase
        df.to_csv(f'{out_dir}/{table}.csv')


def open_all_FF_tables(out_dir='../data/FF_tables', 
                       typo_replacement_dict=config.typo_replacement_dict, 
                       msha_replace_dict=config.msha_replace_dict,
                       msha_replace_by_mine_name=config.msha_replace_by_mine_name, 
                       num_shafts_replacement=config.num_shafts_replacement):
    """ Opens all pre-retrieved data from the FF section of the EPA GHRP. """
    ff_tables = {table:pd.read_csv(f'{out_dir}/{table}.csv', low_memory=False) for table in config.FF_TABLES}
    ff_tables['FF_VENTILATION_QTRLY'] = correct_typos(ff_tables['FF_VENTILATION_QTRLY'], 
                                                      'active_ventilation_start_date', 
                                                      typo_replacement_dict)
    ff_tables['FF_VENTILATION_QTRLY'] = correct_typos(ff_tables['FF_VENTILATION_QTRLY'], 
                                                      'active_ventilation_stop_date', 
                                                      typo_replacement_dict)
    for table in config.FF_TABLES:
        df = ff_tables[table]
        if 'facility_name' in df.columns:
            df['facility_name'] = df['facility_name'].str.upper()
        if 'msha_identification_number' in df.columns:
            df['msha_identification_number'] = clean_up_msha_id(df, msha_replace_dict, msha_replace_by_mine_name)
        if 'num_of_shafts' in df.columns:
            df['num_of_shafts'] = correct_num_shafts(df, num_shafts_replacement)
    return ff_tables


def get_timestamp_from_id(plume_or_swath_ids):
    return pd.to_datetime(
        plume_or_swath_ids.str.replace("GAO|ang|emi|p(.*)|-(.*)", "", regex=True), utc=True
    )


def read_infrastructure_file(infrastructure_filename, mine_replacement_dict=config.mine_replacement_dict):
    if infrastructure_filename.endswith(".kml"):
        vents = gpd.read_file(infrastructure_filename)
        vents["vent_id"] = vents["Name"].apply(
            lambda x: int(x.split("-")[0].strip().replace("ID", ""))
        )
        vents["mine"] = vents["Name"].apply(lambda x: x.split("-")[1].strip())
        vents["vent_type"] = vents["Name"].apply(
            lambda x: x.split("-")[2].strip() if len(x.split("-")) > 2 else None
        )
        vents["lat"] = vents["geometry"].apply(lambda x: x.y)
        vents["lon"] = vents["geometry"].apply(lambda x: x.x)

        vents["mine"] = vents["mine"].str.upper().replace(
            mine_replacement_dict
        )
    elif infrastructure_filename.endswith(".csv"):
        vents = pd.read_csv(infrastructure_filename)
        vents["geometry"] = gpd.points_from_xy(vents.lon, vents.lat)
    else: 
        raise ValueError("File must be .kml or .csv")
    
    return vents

def read_plume_file(filepath):
    """
    Read in plume files from a directory/filepath with wildcards and return a single dataframe.
    """
    plumes = pd.concat((pd.read_csv(f) for f in glob.glob(filepath)), ignore_index=True)
    plumes = plumes.rename(config.PLUME_COLUMN_REPLACEMENT_DICT, axis='columns')
    plumes = plumes.drop_duplicates(subset=["plume_id"])
    if "status" in plumes.columns:
        plumes = plumes[plumes.status!="deleted"]
    if "gas" in plumes.columns:
        plumes = plumes[plumes.gas.str.lower()=="CH4".lower()]
    plumes["geometry"] = gpd.points_from_xy(
            plumes.plume_longitude, plumes.plume_latitude, crs=4326
        )
    return plumes.reset_index()


def subset_within_border(plumes, border, border_name):
    """
    Subset plumes to those within the given border
    """
    is_in_border = plumes.geometry.apply(border.contains).values.squeeze()
    plumes.loc[is_in_border, "state"] = border_name
    return plumes.loc[is_in_border, :] 


def convert_geojson_to_kml(filepath):
    """
    Convert all geojson files in a directory to kml files.

    Args:
        filepath: a string with a wildcard to match all files in a directory
                  e.g. '../data/infrastructure/*.geojson'
    """
    mine_list = glob.glob(filepath)
    for mine in mine_list:
        directory = filepath.split("*")[0]
        gpd.read_file(mine).to_file(f'{directory}{mine.split("/")[-1].split(".")[0]}.kml', driver='KML')


def rma_regression(x_in, y_in):
    """ source: http://stratigrafia.org/8370/lecturenotes/regression.html """
    nanmask = np.isnan(x_in) | np.isnan(y_in) # clean inputs
    x,y = x_in[~nanmask], y_in[~nanmask]
    r = pearsonr(x,y).statistic
    sign = 1 if r>=0 else -1
    slope = sign * np.std(y) / np.std(x)
    intercept = np.mean(y) - np.mean(x) * slope

    return slope, intercept, r

def bootstrap_rma_slope(x, y, n_iterations=1000):
    """Bootstrap standard deviation for the RMA slope"""
    nanmask = np.isnan(x) | np.isnan(y)
    x, y = x[~nanmask], y[~nanmask]
    slopes = []
    for _ in range(n_iterations):
        idx = np.random.choice(len(x), len(x), replace=True)
        bs_x, bs_y = x[idx], y[idx]
        slope, _, _ = rma_regression(bs_x, bs_y)
        slopes.append(slope)
    return np.mean(slopes), np.std(slopes)


def format_sigfigs(val, sigfigs=2):
    """Format a number to specified significant figures, forcing decimal display for clarity."""
    if val == 0:
        return f'0.0'  # handle 0 separately
    from math import log10, floor
    digits = sigfigs - int(floor(log10(abs(val)))) - 1
    formatted = f'{val:.{max(digits, 0)}f}'
    # remove unnecessary trailing zeros after decimal (unless it's needed to show sig figs)
    if '.' in formatted:
        int_part, dec_part = formatted.split('.')
        if len(dec_part.rstrip('0')) == 0:
            return f'{int_part}.0'  # keep one decimal for values like 8.0
        return f'{int_part}.{dec_part.rstrip("0")}'
    return formatted