import geopandas as gpd
import numpy as np
import pandas as pd
import os
import json
import argparse
import download_all_cm_plumes as download

os.chdir(os.path.expanduser("~/coal/jupyter"))
import utils


def average_errors(x):
    return np.sqrt(np.sum(x**2)) / len(x)


average_errors.__name__ = "mean"


def sum_errors(x):
    return np.sqrt(np.sum(x**2))


sum_errors.__name__ = "sum"


def scene_to_date(scenes):
    return pd.to_datetime(
        scenes.str.split("emi|ang|GAO").str[1].str.split("t").str[0], format="%Y%m%d"
    )


def drop_missing_scenes(plumes_per_scene):
    is_missing = plumes_per_scene.observed.isnull()
    if is_missing.any():
        print(
            "These points matched plumes within 150m but no scene.\n"
            "Dropping them because the plume is cut off.\n"
        )
        print(plumes_per_scene[is_missing])
        plumes_per_scene = plumes_per_scene[~plumes_per_scene.observed.isnull()]
    return plumes_per_scene


def match_plumes_to_infrastructure(
    plumes,
    infrastructure,
    max_distance=160,  # meters
    additional_columns=["emission_auto", 
                        "emission_uncertainty_auto"],
):
    """
    Join plumes to the nearest infrastructure point within a certain distance (in meters)
    Args:
        plumes: a dataframe with columns (plume_latitude, plume_longitude, plume_id)
        infrastructure: a dataframe with columns (vent_id, lat, lon) or (vent_id, geometry)
        max_distance: the maximum distance in meters to match a plume to a vent
    """
    infra_gdf = gpd.GeoDataFrame(infrastructure, geometry=infrastructure.geometry, crs=4326).to_crs(
        4087
    )  # meters
    plumes_gdf = gpd.GeoDataFrame(plumes, geometry=plumes.geometry, crs=4326).to_crs(4087)  # meters

    matched_plumes = gpd.sjoin_nearest(
        plumes_gdf,
        infra_gdf,
        how="left",
        max_distance=max_distance,
    ).reset_index(drop=True)

    duplicated_plumes = matched_plumes[matched_plumes.plume_id.duplicated()]
    if np.any(duplicated_plumes):
        raise Warning(
            "These plumes match with more than one item: \n"
            f"plumes: {matched_plumes.plume_id[duplicated_plumes]}\n"
            f"infrastructure: {matched_plumes.vent_id[duplicated_plumes]}"
        )
    matched_plumes["scene_id"] = matched_plumes.plume_id.str.split("-").str[0]

    columns_to_keep = ["plume_latitude", "plume_longitude", "vent_id", "plume_id", "scene_id"]
    return matched_plumes[matched_plumes.vent_id.notnull()][columns_to_keep + additional_columns]


def apply_qc_to_plume_scene(plumes_per_scene, 
                            qc_filepath, 
                            missing_filepath,
                            outfile=None, 
                            save_out=False):
    missing = pd.read_csv(missing_filepath, dtype={"vent_id":int})
    qc = pd.read_csv(qc_filepath)
    plumes_per_scene = plumes_per_scene[~plumes_per_scene.scene_id.str.contains("emi")] # drop EMIT

    missing = missing[missing.qc.isin(["none","nondetect"])] # all other plumes MUST be handled by QC
    plumes_per_scene = pd.merge(plumes_per_scene, missing[['scene_id', 'vent_id', 'qc']], on=["scene_id", "vent_id"], how='left')
    plumes_per_scene = pd.merge(plumes_per_scene, qc[['plume_id', 'qc']], on='plume_id', how='left', suffixes=('_missing', '_qc'))
    plumes_per_scene['qc'] = plumes_per_scene.qc_qc.combine_first(plumes_per_scene.qc_missing) # if plume is not in QC df, use assessemnt from Missing df

    assert not plumes_per_scene.qc.isnull().any(), "nans present in QC files, finish QC'ing and try again."
    print("QC summary:")
    print(plumes_per_scene.qc.value_counts())

    plumes_per_scene["qc"] = plumes_per_scene.qc.replace({"nondetect": "hide", 
                                                "none": "fail", 
                                                }) # standardize QC names

    plumes_per_scene = plumes_per_scene[~(plumes_per_scene.qc=="hide")] # drop hide - it's as if we never observed it. 
    plumes_per_scene.loc[plumes_per_scene.qc=='fail','emission_auto'] = 0. # set fail to 0 - no emissions observed.
    plumes_per_scene.loc[(plumes_per_scene.qc=='fail')&
                    ((plumes_per_scene.scene_id.str.contains('ang'))|
                    (plumes_per_scene.scene_id.str.contains('GAO'))), 
                    'emission_uncertainty_auto'] = 50. # set error to upper level 90% POD
    plumes_per_scene['date'] = pd.to_datetime(plumes_per_scene['date'])

    if save_out:
        plumes_per_scene.to_csv(outfile)
        
    return plumes_per_scene


def calculate_emissions_per_day(
    plumes,
    infrastructure,
    coincident_scenes,
    save_out=False,
    out_dir="./",
    output_filename="emissions_per_day.csv",
    save_plumes_per_scene=False,
    plumes_per_scene_filename="plumes_per_scene.csv",
    plumes_per_scene=None,
    qc_plumes=False,
    qc_filepath=None, 
    missing_filepath=None,
): 

    coincident_scenes["observed"] = True
    coincident_scenes["date"] = scene_to_date(coincident_scenes.scene_id)

    if plumes_per_scene is None:
        matched_plumes = match_plumes_to_infrastructure(
            plumes,
            infrastructure,
            max_distance=160,
            additional_columns=["emission_auto", "emission_uncertainty_auto"],
        )

        plumes_per_scene = drop_missing_scenes(
            pd.merge(coincident_scenes, matched_plumes, on=["vent_id", "scene_id"], how="outer")
        )

    if qc_plumes:
        plumes_per_scene = apply_qc_to_plume_scene(
                            plumes_per_scene=plumes_per_scene, 
                            qc_filepath=qc_filepath, 
                            missing_filepath=missing_filepath,
                            outfile=None, 
                            save_out=False)

    emissions_per_day = (
        plumes_per_scene.groupby(["vent_id", "date"])
        .agg(
            {
                "emission_auto": np.nanmean,
                "emission_uncertainty_auto": average_errors,
                "observed": "any",
                "plume_id": lambda x: ", ".join(x.astype(str).to_list()),
                "scene_id": lambda x: ", ".join(x.astype(str).to_list()),
            }
        )
        .reset_index()
    )
    merge_axis = pd.merge(
        coincident_scenes["vent_id"].drop_duplicates(),
        coincident_scenes["date"].drop_duplicates(),
        how="cross",
    )
    emissions_per_day = pd.merge(merge_axis, emissions_per_day, on=["vent_id", "date"], how="left")
    emissions_per_day.observed = emissions_per_day.observed.fillna(False)

    if save_out:
        emissions_per_day.to_csv(out_dir + output_filename, index=False)
    if save_plumes_per_scene:
        plumes_per_scene.to_csv(out_dir + plumes_per_scene_filename, index=False)

    return emissions_per_day


def main(infrastructure_filepath, 
         plumes_filepath, 
         scenes_filepath,
         out_dir, 
         qc_filepath, 
         missing_filepath, 
         vents_and_wells_only, 
         refresh_plumes, 
         qc_plumes):
    # Load data
    infrastructure = utils.read_infrastructure_file(infrastructure_filepath)
    if refresh_plumes:
        download.download_all_plumes(out_file=plumes_filepath, save_out=True)
    plumes = utils.read_plume_file(plumes_filepath)
    coincident_scenes = pd.read_csv(scenes_filepath)

    if vents_and_wells_only:
        infrastructure = infrastructure[
            infrastructure.vent_type.isin(["Ventilation Shaft", "Well"])
        ]

    # Calculate emissions per day
    calculate_emissions_per_day(
        plumes, 
        infrastructure, 
        coincident_scenes,
        save_out=True, 
        save_plumes_per_scene=True, 
        out_dir=out_dir, 
        qc_filepath=qc_filepath, 
        missing_filepath=missing_filepath, 
        qc_plumes=qc_plumes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infrastructure", default="../data/labeled_vents.kml")
    parser.add_argument("-p", "--plumes", default="../data/All_CarbonMapper_Internal_Plumes.csv")
    parser.add_argument("-s", "--scenes", default="../data/overlapping_scenes.csv")
    parser.add_argument("-o", "--outdir", default="./daily_emis/")
    parser.add_argument("-q", "--qc", default="../data/plume_qc_new_20250217.csv")
    parser.add_argument("-m", "--missing", default="../data/missing_plume_check.csv")
    parser.add_argument("--vents_and_wells_only", action='store_true')
    parser.add_argument("--refresh_plumes", action='store_true')
    parser.add_argument("--qc_plumes", action='store_true')
    if not os.path.exists(parser.parse_args().outdir):
        os.makedirs(parser.parse_args().outdir)
    main(
        infrastructure_filepath=parser.parse_args().infrastructure,
        plumes_filepath=parser.parse_args().plumes,
        scenes_filepath=parser.parse_args().scenes,
        out_dir=parser.parse_args().outdir,
        qc_filepath=parser.parse_args().qc,
        missing_filepath=parser.parse_args().missing,
        vents_and_wells_only=parser.parse_args().vents_and_wells_only,
        refresh_plumes=parser.parse_args().refresh_plumes,
        qc_plumes=parser.parse_args().qc_plumes
    )
