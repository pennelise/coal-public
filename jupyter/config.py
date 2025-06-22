PLUMES_PER_SCENE = f'../data/kelly_plumes_20250211/radial_MAD_alpha/plumes_per_scene.csv'

EMISSIONS_PER_DAY = f'../data/kelly_plumes_20250211/radial_MAD_alpha/emissions_per_day.csv'

FF_TABLES = \
            [
            'FF_SUBPART_LEVEL_INFORMATION', # i think this is the totals in terms of GHG forcing
            'FF_SUMMARY_SOURCE', # I think this is the sum of individual vents/wells for each mine for each quarter.
            'FF_DEGASIFICATION', # Labels for degas wells? 
            'FF_WELL_AND_SHAFT', # Labels for each well. Type (well/shaft), owner, facility, start date, end date, etc. Some wells seem to have the MHSA quarterly sampling point (but no lat/lon)
            'FF_VENTILATION_QTRLY', # includes meth_vent_lib and meth_conc_vent, unclear if flow is reported
            'FF_DEGAS_QTRLY', # has quarterly concentrations calulated weekly and from CEMS, 
            'FF_DESTROFFSITE_WEEKLY', # how much is burned offsite (and looks like onsite too)
            'FF_DEGAS_WEEKLY', # includes how it's monitored. CH4 and Flow Rate.
            'FF_DESTROFFSITE_QTRLY', # same as weekly, but quarterly
            'FF_DESTROFFSITE_BACKUP', # no idea... says if there's a backup to desctruction? Has some more details. 
            ] 

typo_replacement_dict = {  
                    'active_ventilation_start_date':
                        {   '20016': '2016', 
                        },

                    'active_ventilation_stop_date':
                        {   '20117': '2017', 
                            '3015': '2015', 
                            '204': '2014', 
                            '0201-09-30': '2017-09-30', 
                            '0204-09-30': '2014-09-30',
                            '0205-09-30' : '2015-09-30',
                            '0217-03-31' : '2017-03-31',
                        } 
                        }

mine_replacement_dict = {
            "CUMBERLAND": "IRON CUMBERLAND, LLC",
            "EMERALD": "IRON EMERALD, LLC",
            "ENLOW FORK": "ENLOW FORK MINE",
            "BAILEY MINE": "BAILEY MINE-CRABAPPLE PORTAL",
            "TUNNEL RIDGE": "TUNNEL RIDGE, LLC",
            "LEER SOUTH": "LEER SOUTH MINING COMPLEX",
            "PRAIRIE EAGLE" : "PRAIRIE EAGLE MINE",
            "MACH #1 MINE": "POND CREEK NO. 1 MINE",
            "MC#1 MINE" : "SUGAR CAMP ENERGY, LLC", 
            "LIVELY GROVE MINE" : "PRAIRIE STATE GENERATING STATION",
            "MINE NO. 1": "WHITE OAK RESOURCES MINE NO. 1/HAMILTON COUNTY COAL",
            "BUCHANAN MINE": "BUCHANAN MINE #1", 
            "BECKLEY POCAHONTAS": "ICG BECKLEY, LLC",
            "BLACK EAGLE MINE": "BLACK EAGLE DEEP MINE",
            "LOWER WAR EAGLE MINE": "LOWER WAR EAGLE DEEP MINE", 
            "NO 7": "WARRIOR MET COAL, LLC", 
            "MINE NO 4": "WARRIOR MET COAL MINE #4",
            "WEST ELK": "WEST ELK MINE",
        }

msha_replace_dict = {
    '102901': '0102901', # Shoal creek mine
    'M36114890': '4604168', # LEER SOUTH MINING COMPLEX, 
    '4505437': '4605437', # AMERICAN EAGLE A.K.A. SPEED MINE
}

msha_replace_by_mine_name = {
    'CARLISLE MINE' : '1202349',
    'OAKTOWN FUELS MINE NO.1' : '1202394',
    'OAKTOWN FUELS MINE NO. 2' : '1202418',
}

PLUME_COLUMN_REPLACEMENT_DICT = {
    "emission_forecast" : "emission_auto",
    "emission_uncertainty_forecast" : "emission_uncertainty_auto",
    # "sector": "ipcc_sector", 
    # "scene_timestamp": "datetime", 
    }

#  format: (facility_name, reporting_year, quarter, number_of_shafts)
num_shafts_replacement = [("OAK GROVE MINE", 2022, 2, 5),
                          ("OAK GROVE MINE", 2022, 3, 5),
                          ("BAILEY MINE-CRABAPPLE PORTAL", 2022, 4, 7),
                          ("ENLOW FORK MINE", 2022, 4, 5),
                          ("HARVEY MINE", 2022, 4, 3),
                          ("HARRISON COUNTY MINE", 2022, 4, 2),
                          ("MARION COUNTY MINE", 2021, 2, 7),
                          ("MARION COUNTY MINE", 2022, 4, 7),
                          ("MARSHALL COUNTY MINE", 2021, 3, 15),
                          ("MARSHALL COUNTY MINE", 2022, 4, 14),
                          ("MONONGALIA COUNTY MINE", 2021, 2, 7), 
                          ("MONONGALIA COUNTY MINE", 2021, 3, 7), 
                          ("LEER SOUTH MINING COMPLEX", 2022, 4, 4)]