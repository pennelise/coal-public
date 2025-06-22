import pandas as pd
import numpy as np
import fiona
import utils
fiona.drvsupport.supported_drivers['kml'] = 'rw'
fiona.drvsupport.supported_drivers['KML'] = 'rw'

from daily_emis import average_errors, sum_errors
from utils import date_to_year_quarter, open_all_FF_tables


def carbonmapper(emissions_per_day_file, lookup_table_file, vent_file):
        lut = pd.read_csv(lookup_table_file)
        vents = utils.read_infrastructure_file(vent_file)
        emissions_per_day = pd.read_csv(emissions_per_day_file)

        lut['msha_identification_number'] = lut['msha_identification_number'].astype(str).str.replace('.0', '', regex=False)
        lut['facility_id'] = lut['facility_id'].astype(str).str.replace('.0', '', regex=False)
        lut = lut[~lut.facility_id.isin(['1005297'])] # Drop Prep plant for road fork 42, because it duplicates facility_id

        emissions_per_day = pd.merge(emissions_per_day, vents[['vent_id', 'vent_type', 'mine']], on='vent_id', how='left')
        emissions_per_day = pd.merge(emissions_per_day, lut[['facility_name', 'facility_id']], left_on='mine', right_on='facility_name', how='left')

        emissions_per_day.date = pd.to_datetime(emissions_per_day.date)
        emissions_per_day['year_quarter'] = date_to_year_quarter(emissions_per_day.date)
        emissions_per_day['year'] = emissions_per_day.date.dt.year.astype(int)
        emissions_per_day['quarter'] = emissions_per_day.date.dt.quarter.astype(int)
        emissions_per_day['emission_auto'] = emissions_per_day['emission_auto'] # units: kg/hr
        emissions_per_day['emission_uncertainty_auto'] = emissions_per_day['emission_uncertainty_auto'] # units: kg/hr
        emissions_per_day = emissions_per_day[emissions_per_day.observed] # no use for non-observed days

        # aggregate to quarterly
        emissions_per_quarter = emissions_per_day.groupby(['facility_id', 'facility_name', 'vent_id', 'year', 'quarter', 'vent_type']
                                                        ).agg({'emission_auto': np.nanmean, 
                                                                'emission_uncertainty_auto': average_errors,
                                                                'observed': 'sum'}).reset_index()

        mine_emis = emissions_per_quarter[emissions_per_quarter.vent_type.isin(['Ventilation Shaft', 'Well'])].groupby(
                                        ['facility_id', 
                                        'facility_name',
                                        'year', 
                                        'quarter', 
                                        'vent_type']
                                        ).agg({'emission_auto': np.nansum, 
                                                'emission_uncertainty_auto': sum_errors, 
                                                'vent_id': 'nunique',
                                                'observed': 'sum'}).reset_index()

        mine_emis = mine_emis.pivot_table(
                        index=['facility_id', 'facility_name', 'year', 'quarter'], 
                        columns='vent_type', 
                        values=['emission_auto', 'emission_uncertainty_auto', 'vent_id', 'observed']
                        ).reset_index().rename(columns={'Ventilation Shaft': 'vent', 
                                                        'Well': 'well'}).swaplevel(axis=1)
        mine_emis[[('total','emission_auto'), ('total','vent_id'), ('total','observed')]
                ] = mine_emis.loc[:,['vent','well']].groupby(level=1, axis=1
                ).sum(min_count=1)[['emission_auto', 'vent_id', 'observed']]
        mine_emis[('total','emission_uncertainty_auto')
                ] = mine_emis[[('vent', 'emission_uncertainty_auto'),('well', 'emission_uncertainty_auto')]].apply(sum_errors,axis=1)
        mine_emis.columns = mine_emis.columns.map('_'.join).str.strip('_')
        return mine_emis


def ghgrp(): # UNITS: Metric Tons of methane
        epa_data = open_all_FF_tables()

        epa_summary = epa_data['FF_SUMMARY_SOURCE']
        epa_summary['is_gas_coll_system'] = epa_summary['is_gas_coll_system'].replace({'Y': 1, 'N': 0}).astype(float)
        epa_summary['quarter'] = epa_summary.quarter.replace({"QUARTER 1 (JAN-MAR)": 1, "QUARTER 2 (APR-JUN)": 2, "QUARTER 3 (JUL-SEP)": 3, "QUARTER 4 (OCT-DEC)": 4})
        epa_summary['year_quarter'] = epa_summary.reporting_year + (epa_summary.quarter-1)/4
        epa_summary['facility_id'] = epa_summary['facility_id'].astype(str).str.replace('.0', '', regex=False)

        epa_summary['net_well_emis'] = epa_summary['qtr_meth_liber_degas_calc'] - epa_summary['qtr_meth_dest_trns_offste_calc']
        epa_summary.loc[epa_summary['net_well_emis']<=0., 'net_well_emis'] = 0.

        return epa_summary