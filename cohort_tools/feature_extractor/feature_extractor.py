#import required packages
import uuid
import json 
import pandas as pd
from rpy2.robjects import r
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import subprocess
import yaml

from rpy2.robjects import pandas2ri
pandas2ri.activate()
base = importr('base')
feature_extraction = importr('FeatureExtraction')
patient_level_prediction = importr('PatientLevelPrediction')
database_connector = importr('DatabaseConnector')


class FeatureExtractor():
    """
    Class for extracting and transforming data in the OMOP Common Data Model 
    format to build patient-level predictive models using lightsaber
    """
    
    def __init__(self,
                 **kwargs
                 ):
        
        if 'cohort_connector' not in kwargs.keys():
            print('please specify cohort_connector')
        elif len(kwargs.keys()) < 2:
            print("""Declare feature extraction parameters or specify path to json file with feature extraction settings """)
        else:
            self.__analysis_id__()
            self.__cohort_connector(kwargs['cohort_connector'])
            settings = self.__load_settings__(**kwargs)
            self.__analysis_name__(settings, **kwargs)
            self.__working_directory__(settings, **kwargs)
            self.__output_directory__(settings, **kwargs)
            self.__covariate__settings__(settings, **kwargs)
            self.__model_training_settings__(settings, **kwargs)
            self.__expt_config_settings__(settings, **kwargs)
            print('Successfully created all settings')
                

    def __cohort_connector(self, cohort_connector):
        try:
            self._db_connection_details = cohort_connector.db_connection_details
            self._r_db_connection_details = cohort_connector.r_db_connection_details
            self._cohort_details = cohort_connector.cohort_details
        except:
            print('Error missing or invalid cohort_connector')
            
            
    def __load_settings__(self, **kwargs): 
        
        settings = {}
        try:
            with open(kwargs['file_path'], 'r') as f:
                try:
                    settings = json.load(f) 
                except ValueError:
                    print("Invalid JSON in " + kwargs['file_path'])
        except KeyError:
            print("file_path parameter not found")
            
        except OSError:
            print("Could not open " + kwargs['file_path'])
        
        else:
            if settings:
                settings['model_training_settings']['sample_size'] = settings['model_training_settings']['sample_size'] if settings['model_training_settings']['sample_size'] else r('NULL')
                settings['model_training_settings']['random_state'] = settings['model_training_settings']['random_state'] if 'random_state' in settings['model_training_settings'].keys() else None
            return settings

                 
     
    def __analysis_id__(self):
        print('Setting analysis id')
        self._analysis_id = str(uuid.uuid4())
    
    @property
    def analysis_id(self):
        return self._analysis_id
    
    
        
    def __analysis_name__(self, settings, **kwargs):
        try:
            print('Setting analysis name')
            if settings:
                self._analysis_name = settings['analysis_name']
            else:
                self._analysis_name = kwargs['analysis_name']
        except:
            print("""
                  Missing/incorrect analysis_name
            """)

    @property
    def analysis_name():
        return self._analysis_name
  
   
    
    
    def __working_directory__(self, settings, **kwargs):
        try:
            print('Setting working directory')
            if settings:
                self._working_directory = settings['working_directory']
            else:
                self._working_directory = kwargs['working_directory']
            if not os.path.exists(self._working_directory):
                os.makedirs(self._working_directory)
        except:
            print("""
                  Missing/incorrect working_directory
            """)

    @property
    def working_directory():
        return self._working_directory
    
    
    def __output_directory__(self, settings, **kwargs):
        try:
            print('Setting output directory')
            if settings:
                self._output_directory = settings['output_directory']
            else:
                self._output_directory = kwargs['output_directory']
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory)
        except:
            print("""
                  Missing/incorrect output_directory
            """)

    @property
    def output_directory():
        return self._output_directory
    
    
    def __covariate__settings__(self, settings, **kwargs):
        
        self.__baseline_covariate_setting_options__(settings, **kwargs)
        self.__baseline_covariate_settings__()
       
        self.__tidy_plp_settings__(settings, **kwargs)
        
        self.__temporal_covariate_setting_options__(settings, **kwargs) 
        self.__temporal_covariate_settings__()
        
        
    def __baseline_covariate_setting_options__(self, settings, **kwargs):
        options = { 'use_demographics_gender' : False,
                    'use_demographics_age' : False,
                    'use_demographics_age_group' : False,
                    'use_demographics_race' : False,
                    'use_demographics_ethnicity' : False,
                    'use_demographics_index_year' : False,
                    'use_demographics_index_month' : False,
                    'use_demographics_prior_observation_time' : False,
                    'use_demographics_post_observation_time' : False,
                    'use_demographics_time_in_cohort' : False,
                    'use_demographics_index_year_month' : False,
                    'use_condition_occurrence_any_time_prior' : True,
                    'use_condition_occurrence_long_term' : False,
                    'use_condition_occurrence_medium_term' : False,
                    'use_condition_occurrence_short_term' : False,
                    'use_condition_occurrence_primary_inpatient_any_time_prior' : False,
                    'use_condition_occurrence_primary_inpatient_long_term' : False,
                    'use_condition_occurrence_primary_inpatient_medium_term' : False,
                    'use_condition_occurrence_primary_inpatient_short_term' : False,
                    'use_condition_era_any_time_prior' : False,
                    'use_condition_era_long_term' : False,
                    'use_condition_era_medium_term' : False,
                    'use_condition_era_short_term' : False,
                    'use_condition_era_overlapping' : False,
                    'use_condition_era_start_long_term' : False,
                    'use_condition_era_start_medium_term' : False,
                    'use_condition_era_start_short_term' : False,
                    'use_condition_group_era_any_time_prior' : False,
                    'use_condition_group_era_long_term' : False,
                    'use_condition_group_era_medium_term' : False,
                    'use_condition_group_era_short_term' : False,
                    'use_condition_group_era_overlapping' : False,
                    'use_condition_group_era_start_long_term' : False,
                    'use_condition_group_era_start_medium_term' : False,
                    'use_condition_group_era_start_short_term' : False,
                    'use_drug_exposure_any_time_prior' : False,
                    'use_drug_exposure_long_term' : False,
                    'use_drug_exposure_medium_term' : False,
                    'use_drug_exposure_short_term' : False,
                    'use_drug_era_any_time_prior' : False,
                    'use_drug_era_long_term' : False,
                    'use_drug_era_medium_term' : False,
                    'use_drug_era_short_term' : False,
                    'use_drug_era_overlapping' : False,
                    'use_drug_era_start_long_term' : False,
                    'use_drug_era_start_medium_term' : False,
                    'use_drug_era_start_short_term' : False,
                    'use_drug_group_era_any_time_prior' : False,
                    'use_drug_group_era_long_term' : False,
                    'use_drug_group_era_medium_term' : False,
                    'use_drug_group_era_short_term' : False,
                    'use_drug_group_era_overlapping' : False,
                    'use_drug_group_era_start_long_term' : False,
                    'use_drug_group_era_start_medium_term' : False,
                    'use_drug_group_era_start_short_term' : False,
                    'use_procedure_occurrence_any_time_prior' : False,
                    'use_procedure_occurrence_long_term' : False,
                    'use_procedure_occurrence_medium_term' : False,
                    'use_procedure_occurrence_short_term' : False,
                    'use_device_exposure_any_time_prior' : False,
                    'use_device_exposure_long_term' : False,
                    'use_device_exposure_medium_term' : False,
                    'use_device_exposure_short_term' : False,
                    'use_measurement_any_time_prior' : False,
                    'use_measurement_long_term' : False,
                    'use_measurement_medium_term' : False,
                    'use_measurement_short_term' : False,
                    'use_measurement_value_any_time_prior' : False,
                    'use_measurement_value_long_term' : False,
                    'use_measurement_value_medium_term' : False,
                    'use_measurement_value_short_term' : False,
                    'use_measurement_range_group_any_time_prior' : False,
                    'use_measurement_range_group_long_term' : False,
                    'use_measurement_range_group_medium_term' : False,
                    'use_measurement_range_group_short_term' : False,
                    'use_observation_any_time_prior' : False,
                    'use_observation_long_term' : False,
                    'use_observation_medium_term' : False,
                    'use_observation_short_term' : False,
                    'use_charlson_index' : False,
                    'use_dcsi' : False,
                    'use_chads2' : False,
                    'use_chads2_vasc' : False,
                    'use_hfrs' : False,
                    'use_distinct_condition_count_long_term' : False,
                    'use_distinct_condition_count_medium_term' : False,
                    'use_distinct_condition_count_short_term' : False,
                    'use_distinct_ingredient_count_long_term' : False,
                    'use_distinct_ingredient_count_medium_term' : False,
                    'use_distinct_ingredient_count_short_term' : False,
                    'use_distinct_procedure_count_long_term' : False,
                    'use_distinct_procedure_count_medium_term' : False,
                    'use_distinct_procedure_count_short_term' : False,
                    'use_distinct_measurement_count_long_term' : False,
                    'use_distinct_measurement_count_medium_term' : False,
                    'use_distinct_measurement_count_short_term' : False,
                    'use_distinct_observation_count_long_term' : False,
                    'use_distinct_observation_count_medium_term' : False,
                    'use_distinct_observation_count_short_term' : False,
                    'use_visit_count_long_term' : False,
                    'use_visit_count_medium_term' : False,
                    'use_visit_count_short_term' : False,
                    'use_visit_concept_count_long_term' : False,
                    'use_visit_concept_count_medium_term' : False,
                    'use_visit_concept_count_short_term' : False,
                    'long_term_start_days' : -365,
                    'medium_term_start_days' : -180,
                    'short_term_start_days' : -30,
                    'end_days' : 0.0,
                    'included_covariate_concept_ids' : [],
                    'add_descendants_to_include' : False,
                    'excluded_covariate_concept_ids' : [],
                    'add_descendants_to_exclude' : False,
                    'included_covariate_ids' : []
                  }
        
        
        try:
            print("""Setting baseline covariate options""") 
            if settings:
                options.update(dict((k, settings['covariate_settings'][k]) for k in options.keys() if k in settings['covariate_settings'].keys()))
            else:
                options.update(dict((k, kwargs[k]) for k in options.keys() if k in kwargs.keys()))

            self._baseline_covariate_setting_options = options 
        
        except:
            print("""Error in setting baseline covariate options""") 
    
    @property
    def baseline_covariate_setting_options(self):
        return self._baseline_covariate_setting_options 
  


    def __baseline_covariate_settings__(self):
        """
        Creates an object of type covariateSettings for baseline covariates, to be used in other functions.
        """  
        try:
            print('Constructing baseline covariate settings')
            self._baseline_covariate_settings  = feature_extraction.createCovariateSettings(
                                                useDemographicsGender = self._baseline_covariate_setting_options['use_demographics_gender'],
                                                useDemographicsAge = self._baseline_covariate_setting_options['use_demographics_age'],
                                                useDemographicsAgeGroup = self._baseline_covariate_setting_options['use_demographics_age_group'],
                                                useDemographicsRace = self._baseline_covariate_setting_options['use_demographics_race'],
                                                useDemographicsEthnicity = self._baseline_covariate_setting_options['use_demographics_ethnicity'],
                                                useDemographicsIndexYear = self._baseline_covariate_setting_options['use_demographics_index_year'],
                                                useDemographicsIndexMonth = self._baseline_covariate_setting_options['use_demographics_index_month'],
                                                useDemographicsPriorObservationTime = self._baseline_covariate_setting_options['use_demographics_prior_observation_time'],
                                                useDemographicsPostObservationTime = self._baseline_covariate_setting_options['use_demographics_post_observation_time'],
                                                useDemographicsTimeInCohort = self._baseline_covariate_setting_options['use_demographics_time_in_cohort'],
                                                useDemographicsIndexYearMonth = self._baseline_covariate_setting_options['use_demographics_index_year_month'],
                                                useConditionOccurrenceAnyTimePrior = self._baseline_covariate_setting_options['use_condition_occurrence_any_time_prior'],
                                                useConditionOccurrenceLongTerm = self._baseline_covariate_setting_options['use_condition_occurrence_long_term'],
                                                useConditionOccurrenceMediumTerm = self._baseline_covariate_setting_options['use_condition_occurrence_medium_term'],
                                                useConditionOccurrenceShortTerm = self._baseline_covariate_setting_options['use_condition_occurrence_short_term'],
                                                useConditionOccurrencePrimaryInpatientAnyTimePrior = self._baseline_covariate_setting_options['use_condition_occurrence_primary_inpatient_any_time_prior'],
                                                useConditionOccurrencePrimaryInpatientLongTerm = self._baseline_covariate_setting_options['use_condition_occurrence_primary_inpatient_long_term'],
                                                useConditionOccurrencePrimaryInpatientMediumTerm = self._baseline_covariate_setting_options['use_condition_occurrence_primary_inpatient_medium_term'],
                                                useConditionOccurrencePrimaryInpatientShortTerm = self._baseline_covariate_setting_options['use_condition_occurrence_primary_inpatient_short_term'],
                                                useConditionEraAnyTimePrior = self._baseline_covariate_setting_options['use_condition_era_any_time_prior'],
                                                useConditionEraLongTerm = self._baseline_covariate_setting_options['use_condition_era_long_term'],
                                                useConditionEraMediumTerm = self._baseline_covariate_setting_options['use_condition_era_medium_term'],
                                                useConditionEraShortTerm = self._baseline_covariate_setting_options['use_condition_era_short_term'],
                                                useConditionEraOverlapping = self._baseline_covariate_setting_options['use_condition_era_overlapping'],
                                                useConditionEraStartLongTerm = self._baseline_covariate_setting_options['use_condition_era_start_long_term'],
                                                useConditionEraStartMediumTerm = self._baseline_covariate_setting_options['use_condition_era_start_medium_term'],
                                                useConditionEraStartShortTerm = self._baseline_covariate_setting_options['use_condition_era_start_short_term'],
                                                useConditionGroupEraAnyTimePrior = self._baseline_covariate_setting_options['use_condition_group_era_any_time_prior'],
                                                useConditionGroupEraLongTerm = self._baseline_covariate_setting_options['use_condition_group_era_long_term'],
                                                useConditionGroupEraMediumTerm = self._baseline_covariate_setting_options['use_condition_group_era_medium_term'],
                                                useConditionGroupEraShortTerm = self._baseline_covariate_setting_options['use_condition_group_era_short_term'],
                                                useConditionGroupEraOverlapping = self._baseline_covariate_setting_options['use_condition_group_era_overlapping'],
                                                useConditionGroupEraStartLongTerm = self._baseline_covariate_setting_options['use_condition_group_era_start_long_term'],
                                                useConditionGroupEraStartMediumTerm = self._baseline_covariate_setting_options['use_condition_group_era_start_medium_term'],
                                                useConditionGroupEraStartShortTerm = self._baseline_covariate_setting_options['use_condition_group_era_start_short_term'],
                                                useDrugExposureAnyTimePrior = self._baseline_covariate_setting_options['use_drug_exposure_any_time_prior'],
                                                useDrugExposureLongTerm = self._baseline_covariate_setting_options['use_drug_exposure_long_term'],
                                                useDrugExposureMediumTerm = self._baseline_covariate_setting_options['use_drug_exposure_medium_term'],
                                                useDrugExposureShortTerm = self._baseline_covariate_setting_options['use_drug_exposure_short_term'],
                                                useDrugEraAnyTimePrior = self._baseline_covariate_setting_options['use_drug_era_any_time_prior'],
                                                useDrugEraLongTerm = self._baseline_covariate_setting_options['use_drug_era_long_term'],
                                                useDrugEraMediumTerm = self._baseline_covariate_setting_options['use_drug_era_medium_term'],
                                                useDrugEraShortTerm = self._baseline_covariate_setting_options['use_drug_era_short_term'],
                                                useDrugEraOverlapping = self._baseline_covariate_setting_options['use_drug_era_overlapping'],
                                                useDrugEraStartLongTerm = self._baseline_covariate_setting_options['use_drug_era_start_long_term'],
                                                useDrugEraStartMediumTerm = self._baseline_covariate_setting_options['use_drug_era_start_medium_term'],
                                                useDrugEraStartShortTerm = self._baseline_covariate_setting_options['use_drug_era_start_short_term'],
                                                useDrugGroupEraAnyTimePrior = self._baseline_covariate_setting_options['use_drug_group_era_any_time_prior'],
                                                useDrugGroupEraLongTerm = self._baseline_covariate_setting_options['use_drug_group_era_long_term'],
                                                useDrugGroupEraMediumTerm = self._baseline_covariate_setting_options['use_drug_group_era_medium_term'],
                                                useDrugGroupEraShortTerm = self._baseline_covariate_setting_options['use_drug_group_era_short_term'],
                                                useDrugGroupEraOverlapping = self._baseline_covariate_setting_options['use_drug_group_era_overlapping'],
                                                useDrugGroupEraStartLongTerm = self._baseline_covariate_setting_options['use_drug_group_era_start_long_term'],
                                                useDrugGroupEraStartMediumTerm = self._baseline_covariate_setting_options['use_drug_group_era_start_medium_term'],
                                                useDrugGroupEraStartShortTerm = self._baseline_covariate_setting_options['use_drug_group_era_start_short_term'],
                                                useProcedureOccurrenceAnyTimePrior = self._baseline_covariate_setting_options['use_procedure_occurrence_any_time_prior'],
                                                useProcedureOccurrenceLongTerm = self._baseline_covariate_setting_options['use_procedure_occurrence_long_term'],
                                                useProcedureOccurrenceMediumTerm = self._baseline_covariate_setting_options['use_procedure_occurrence_medium_term'],
                                                useProcedureOccurrenceShortTerm = self._baseline_covariate_setting_options['use_procedure_occurrence_short_term'],
                                                useDeviceExposureAnyTimePrior = self._baseline_covariate_setting_options['use_device_exposure_any_time_prior'],
                                                useDeviceExposureLongTerm = self._baseline_covariate_setting_options['use_device_exposure_long_term'],
                                                useDeviceExposureMediumTerm = self._baseline_covariate_setting_options['use_device_exposure_medium_term'],
                                                useDeviceExposureShortTerm = self._baseline_covariate_setting_options['use_device_exposure_short_term'],
                                                useMeasurementAnyTimePrior = self._baseline_covariate_setting_options['use_measurement_any_time_prior'],
                                                useMeasurementLongTerm = self._baseline_covariate_setting_options['use_measurement_long_term'],
                                                useMeasurementMediumTerm = self._baseline_covariate_setting_options['use_measurement_medium_term'],
                                                useMeasurementShortTerm = self._baseline_covariate_setting_options['use_measurement_short_term'],
                                                useMeasurementValueAnyTimePrior = self._baseline_covariate_setting_options['use_measurement_value_any_time_prior'],
                                                useMeasurementValueLongTerm = self._baseline_covariate_setting_options['use_measurement_value_long_term'],
                                                useMeasurementValueMediumTerm = self._baseline_covariate_setting_options['use_measurement_value_medium_term'],
                                                useMeasurementValueShortTerm = self._baseline_covariate_setting_options['use_measurement_value_short_term'],
                                                useMeasurementRangeGroupAnyTimePrior = self._baseline_covariate_setting_options['use_measurement_range_group_any_time_prior'],
                                                useMeasurementRangeGroupLongTerm = self._baseline_covariate_setting_options['use_measurement_range_group_long_term'],
                                                useMeasurementRangeGroupMediumTerm = self._baseline_covariate_setting_options['use_measurement_range_group_medium_term'],
                                                useMeasurementRangeGroupShortTerm = self._baseline_covariate_setting_options['use_measurement_range_group_short_term'],
                                                useObservationAnyTimePrior = self._baseline_covariate_setting_options['use_observation_any_time_prior'],
                                                useObservationLongTerm = self._baseline_covariate_setting_options['use_observation_long_term'],
                                                useObservationMediumTerm = self._baseline_covariate_setting_options['use_observation_medium_term'],
                                                useObservationShortTerm = self._baseline_covariate_setting_options['use_observation_short_term'],
                                                useCharlsonIndex = self._baseline_covariate_setting_options['use_charlson_index'],
                                                useDcsi = self._baseline_covariate_setting_options['use_dcsi'],
                                                useChads2 = self._baseline_covariate_setting_options['use_chads2'],
                                                useChads2Vasc = self._baseline_covariate_setting_options['use_chads2_vasc'],
                                                useHfrs = self._baseline_covariate_setting_options['use_hfrs'],
                                                useDistinctConditionCountLongTerm = self._baseline_covariate_setting_options['use_distinct_condition_count_long_term'],
                                                useDistinctConditionCountMediumTerm = self._baseline_covariate_setting_options['use_distinct_condition_count_medium_term'],
                                                useDistinctConditionCountShortTerm = self._baseline_covariate_setting_options['use_distinct_condition_count_short_term'],
                                                useDistinctIngredientCountLongTerm = self._baseline_covariate_setting_options['use_distinct_ingredient_count_long_term'],
                                                useDistinctIngredientCountMediumTerm = self._baseline_covariate_setting_options['use_distinct_ingredient_count_medium_term'],
                                                useDistinctIngredientCountShortTerm = self._baseline_covariate_setting_options['use_distinct_ingredient_count_short_term'],
                                                useDistinctProcedureCountLongTerm = self._baseline_covariate_setting_options['use_distinct_procedure_count_long_term'],
                                                useDistinctProcedureCountMediumTerm = self._baseline_covariate_setting_options['use_distinct_procedure_count_medium_term'],
                                                useDistinctProcedureCountShortTerm = self._baseline_covariate_setting_options['use_distinct_procedure_count_short_term'],
                                                useDistinctMeasurementCountLongTerm = self._baseline_covariate_setting_options['use_distinct_measurement_count_long_term'],
                                                useDistinctMeasurementCountMediumTerm = self._baseline_covariate_setting_options['use_distinct_measurement_count_medium_term'],
                                                useDistinctMeasurementCountShortTerm = self._baseline_covariate_setting_options['use_distinct_measurement_count_short_term'],
                                                useDistinctObservationCountLongTerm = self._baseline_covariate_setting_options['use_distinct_observation_count_long_term'],
                                                useDistinctObservationCountMediumTerm = self._baseline_covariate_setting_options['use_distinct_observation_count_medium_term'],
                                                useDistinctObservationCountShortTerm = self._baseline_covariate_setting_options['use_distinct_observation_count_short_term'],
                                                useVisitCountLongTerm = self._baseline_covariate_setting_options['use_visit_count_long_term'],
                                                useVisitCountMediumTerm = self._baseline_covariate_setting_options['use_visit_count_medium_term'],
                                                useVisitCountShortTerm = self._baseline_covariate_setting_options['use_visit_count_short_term'],
                                                useVisitConceptCountLongTerm = self._baseline_covariate_setting_options['use_visit_concept_count_long_term'],
                                                useVisitConceptCountMediumTerm = self._baseline_covariate_setting_options['use_visit_concept_count_medium_term'],
                                                useVisitConceptCountShortTerm = self._baseline_covariate_setting_options['use_visit_concept_count_short_term'],
                                                longTermStartDays = self._baseline_covariate_setting_options['long_term_start_days'],
                                                mediumTermStartDays = self._baseline_covariate_setting_options['medium_term_start_days'],
                                                shortTermStartDays = self._baseline_covariate_setting_options['short_term_start_days'],
                                                endDays = self._baseline_covariate_setting_options['end_days'],
                                                includedCovariateConceptIds = ro.vectors.IntVector(self._baseline_covariate_setting_options['included_covariate_concept_ids']),
                                                addDescendantsToInclude = self._baseline_covariate_setting_options['add_descendants_to_include'],
                                                excludedCovariateConceptIds = ro.vectors.IntVector(self._baseline_covariate_setting_options['excluded_covariate_concept_ids']),
                                                addDescendantsToExclude = self._baseline_covariate_setting_options['add_descendants_to_exclude'],
                                                includedCovariateIds = ro.vectors.IntVector(self._baseline_covariate_setting_options['included_covariate_ids']))
        except:
            print("""
                  Error in constructing baseline covariate settings
                  """)

    @property
    def baseline_covariate_settings(self):
        return self._baseline_covariate_settings
    

    
    def __tidy_plp_settings__(self, settings, **kwargs):
        
        try:
            print("""Setting covariate tyding options""") 
            if settings:
                self._tidy_plp_settings = settings['tidy_covariate_settings']
            else:
                self._tidy_plp_settings ={'min_fraction' : kwargs['min_fraction'],
                                          'normalize' : kwargs['normalize'],
                                          'remove_redundancy' : kwargs['remove_redundancy']
                                 }
        except:
            print("""Missing/incorrect covariate tyding settings. Specify covariate tidying parameters as follows:
                     min_fraction: Minimum fraction of the population that should have a non-zero value for a covariate
                     normalize: If true, normalize the covariates by dividing by the max
                     remove_redundancy: If true, remove redundant covariates
                    """)
          
    @property
    def tidy_plp_settings(self):
        return self._tidy_plp_settings 
    
    
    def __temporal_covariate_setting_options__(self, settings, **kwargs):
        options = {
                    'use_demographics_gender' : False,
                    'use_demographics_age' : False,
                    'use_demographics_age_group' : False,
                    'use_demographics_race' : False,
                    'use_demographics_ethnicity' : False,
                    'use_demographics_index_year' : False,
                    'use_demographics_index_month' : False,
                    'use_demographics_prior_observation_time' : False,
                    'use_demographics_post_observation_time' : False,
                    'use_demographics_time_in_cohort' : False,
                    'use_demographics_index_year_month' : False,
                    'use_condition_occurrence' : False,
                    'use_condition_occurrence_primary_inpatient' : False,
                    'use_condition_era_start' : False,
                    'use_condition_era_overlap' : False,
                    'use_condition_era_group_start' : False,
                    'use_condition_era_group_overlap' : False,
                    'use_drug_exposure' : False,
                    'use_drug_era_start' : False,
                    'use_drug_era_overlap' : False,
                    'use_drug_era_group_start' : False,
                    'use_drug_era_group_overlap' : False,
                    'use_procedure_occurrence' : False,
                    'use_device_exposure' : False,
                    'use_measurement' : False,
                    'use_measurement_value' : False,
                    'use_measurement_range_group' : False,
                    'use_observation' : False,
                    'use_charlson_index' : False,
                    'use_dcsi' : False,
                    'use_chads2' : False,
                    'use_chads2_vasc' : False,
                    'use_hfrs' : False,
                    'use_distinct_condition_count' : False,
                    'use_distinct_ingredient_count' : False,
                    'use_distinct_procedure_count' : False,
                    'use_distinct_measurement_count' : False,
                    'use_distinct_observation_count' : False,
                    'use_visit_count' : False,
                    'use_visit_concept_count' : False,
                    'temporal_start_days' : list(range(-365,0)),
                    'temporal_end_days' : list(range(-365,0)),
                    'included_covariate_concept_ids' : [],
                    'add_descendants_to_include' : False,
                    'excluded_covariate_concept_ids' : [],
                    'add_descendants_to_exclude' : False,
                    'included_covariate_ids' : []
                  }
        
        try:
            print("""Setting temporal covariate options""") 
            if settings:
                options.update(dict((k, settings['covariate_settings'][k]) for k in options.keys() if k in settings['covariate_settings'].keys()))
            else:
                options.update(dict((k, kwargs[k]) for k in options.keys() if k in kwargs.keys()))

            self._temporal_covariate_setting_options = options            
        except:
            print("""Error in setting baseline covariate options""") 
            
    
    @property
    def temporal_covariate_setting_options(self):
        return self._temporal_covariate_setting_options
    
    
    def __temporal_covariate_settings__(self):
        """
        Creates an object of type covariateSettings for temporal covariates, to be used in other functions.
        """
        try:
            print('Constructing temporal covariate settings')
            self._temporal_covariate_settings = feature_extraction.createTemporalCovariateSettings (
                                                    useDemographicsGender = self._temporal_covariate_setting_options['use_demographics_gender'],
                                                    useDemographicsAge = self._temporal_covariate_setting_options['use_demographics_age'],
                                                    useDemographicsAgeGroup = self._temporal_covariate_setting_options['use_demographics_age_group'],
                                                    useDemographicsRace = self._temporal_covariate_setting_options['use_demographics_race'],
                                                    useDemographicsEthnicity = self._temporal_covariate_setting_options['use_demographics_ethnicity'],
                                                    useDemographicsIndexYear = self._temporal_covariate_setting_options['use_demographics_index_year'],
                                                    useDemographicsIndexMonth = self._temporal_covariate_setting_options['use_demographics_index_month'],
                                                    useDemographicsPriorObservationTime = self._temporal_covariate_setting_options['use_demographics_prior_observation_time'],
                                                    useDemographicsPostObservationTime = self._temporal_covariate_setting_options['use_demographics_post_observation_time'],
                                                    useDemographicsTimeInCohort = self._temporal_covariate_setting_options['use_demographics_time_in_cohort'],
                                                    useDemographicsIndexYearMonth = self._temporal_covariate_setting_options['use_demographics_index_year_month'],
                                                    useConditionOccurrence = self._temporal_covariate_setting_options['use_condition_occurrence'],
                                                    useConditionOccurrencePrimaryInpatient = self._temporal_covariate_setting_options['use_condition_occurrence_primary_inpatient'],
                                                    useConditionEraStart = self._temporal_covariate_setting_options['use_condition_era_start'],
                                                    useConditionEraOverlap = self._temporal_covariate_setting_options['use_condition_era_overlap'],
                                                    useConditionEraGroupStart = self._temporal_covariate_setting_options['use_condition_era_group_start'],
                                                    useConditionEraGroupOverlap = self._temporal_covariate_setting_options['use_condition_era_group_overlap'],
                                                    useDrugExposure = self._temporal_covariate_setting_options['use_drug_exposure'],
                                                    useDrugEraStart = self._temporal_covariate_setting_options['use_drug_era_start'],
                                                    useDrugEraOverlap = self._temporal_covariate_setting_options['use_drug_era_overlap'],
                                                    useDrugEraGroupStart = self._temporal_covariate_setting_options['use_drug_era_group_start'],
                                                    useDrugEraGroupOverlap = self._temporal_covariate_setting_options['use_drug_era_group_overlap'],
                                                    useProcedureOccurrence = self._temporal_covariate_setting_options['use_procedure_occurrence'],
                                                    useDeviceExposure = self._temporal_covariate_setting_options['use_device_exposure'],
                                                    useMeasurement = self._temporal_covariate_setting_options['use_measurement'],
                                                    useMeasurementValue = self._temporal_covariate_setting_options['use_measurement_value'],
                                                    useMeasurementRangeGroup = self._temporal_covariate_setting_options['use_measurement_range_group'],
                                                    useObservation = self._temporal_covariate_setting_options['use_observation'],
                                                    useCharlsonIndex = self._temporal_covariate_setting_options['use_charlson_index'],
                                                    useDcsi = self._temporal_covariate_setting_options['use_dcsi'],
                                                    useChads2 = self._temporal_covariate_setting_options['use_chads2'],
                                                    useChads2Vasc = self._temporal_covariate_setting_options['use_chads2_vasc'],
                                                    useHfrs = self._temporal_covariate_setting_options['use_hfrs'],
                                                    useDistinctConditionCount = self._temporal_covariate_setting_options['use_distinct_condition_count'],
                                                    useDistinctIngredientCount = self._temporal_covariate_setting_options['use_distinct_ingredient_count'],
                                                    useDistinctProcedureCount = self._temporal_covariate_setting_options['use_distinct_procedure_count'],
                                                    useDistinctMeasurementCount = self._temporal_covariate_setting_options['use_distinct_measurement_count'],
                                                    useDistinctObservationCount = self._temporal_covariate_setting_options['use_distinct_observation_count'],
                                                    useVisitCount = self._temporal_covariate_setting_options['use_visit_count'],
                                                    useVisitConceptCount = self._temporal_covariate_setting_options['use_visit_concept_count'],
                                                    temporalStartDays = ro.vectors.IntVector(self._temporal_covariate_setting_options['temporal_start_days']),
                                                    temporalEndDays = ro.vectors.IntVector(self._temporal_covariate_setting_options['temporal_end_days']),
                                                    includedCovariateConceptIds = ro.vectors.IntVector(self._temporal_covariate_setting_options['included_covariate_concept_ids']),
                                                    addDescendantsToInclude = self._temporal_covariate_setting_options['add_descendants_to_include'],
                                                    excludedCovariateConceptIds = ro.vectors.IntVector(self._temporal_covariate_setting_options['excluded_covariate_concept_ids']),
                                                    addDescendantsToExclude = self._temporal_covariate_setting_options['add_descendants_to_exclude'],
                                                    includedCovariateIds = ro.vectors.IntVector(self._temporal_covariate_setting_options['included_covariate_ids']))

        except:
            print("""
                  Error in constructing temporal covariate settings
                  """)                                            


    @property
    def temporal_covariate_settings(self):
        return self._temporal_covariate_settings

    
    
    
    def __model_training_settings__(self, settings, **kwargs):
        try:
            print("""Setting model training options""")
            if settings:
                self._model_training_settings = settings['model_training_settings']
            else:
                self._model_training_settings ={ 
                                            'sample_size': kwargs['sample_size'] if kwargs['sample_size'] else r('NULL'), 
                                            'val_size': kwargs['val_size'], 
                                            'random_state': kwargs['random_state'] if 'random_state' in kwargs.keys() else None,
                                           }          
            
        except:
            print("""Missing/incorrect model training settings. Specify model training parameters as follows:
                     sample_size: The sample size to be extracted from members of the cohort
                     val_size: The proportion of data to used for the training/validation split
                     random_state: Number to control shuffling applied to the data before applying the split
                     path: path for saving extracted data
            """)

    @property
    def model_training_settings(self):
        return self._model_training_settings
    
    
    def __expt_config_settings__(self, settings, **kwargs):
        try:
            print("""Setting experiment config options""")
            if settings:
                self._expt_config_settings = settings['expt_config_settings']
            else:
                self._expt_config_settings = {
                    'categorical_covariate_concept_ids': kwargs['categorical_covariate_concept_ids'],
                    'numerical_covariate_concept_ids': kwargs['numerical_covariate_concept_ids'],
                    'categorical_covariate_concept_value_mappings': kwargs['categorical_covariate_concept_value_mappings'],
                    'normal_covariate_concept_values': kwargs['normal_covariate_concept_values']
                }
        except:
            print("""Missing/incorrect experiment config options. Specify experiment config parameters as follows:
                     categorical_covariate_concept_ids: list of categorical covariate concept identifiers from OMOP CDM 
                     numerical_covariate_concept_ids: list of numerical covariate concept identifiers from OMOP CDM 
                     categorical_covariate_concept_value_mappings: dictionary of concept value mappings. Each key is a concept identifier from OMOP CDM and each value is a dictionary with feature value replacemnt mappings
                     normal_covariate_concept_values: user specified normal values for each concept
            """)
            
    @property
    def expt_config_settings(self):
        return self._expt_config_settings
    

    
    def __get_plp_data__ (self):
        """
        Gets patient level prediction data, an R object of type plpData, containing information on the cohorts, their outcomes, and baseline covariates
        """  
        print('Fetching plpData')
        plp_data = patient_level_prediction.getPlpData( 
                                connectionDetails = self._r_db_connection_details,
                                cdmDatabaseSchema = self._cohort_details['cdm_database_schema'],
                                oracleTempSchema  = self._cohort_details['oracle_temp_schema'],
                                cohortDatabaseSchema = self._cohort_details['target_cohort_database_schema'],
                                cohortTable = self._cohort_details['target_cohort_table'],
                                cohortId = self._cohort_details['target_cohort_id'],
                                outcomeDatabaseSchema =  self._cohort_details['outcome_cohort_database_schema'],
                                outcomeTable = self._cohort_details['outcome_cohort_table'],
                                outcomeIds = self._cohort_details['outcome_cohort_id'],
                                sampleSize = self._model_training_settings['sample_size'],
                                covariateSettings = self._baseline_covariate_settings
                                )
        return plp_data
        
        
      
    
    def __tidy_plp_covariates__(self, plp_data):
        """
        Removes infrequent covariates, normalize, and remove redundancy
        """
        print('Tidying plp covariates')
        r("""
              tidyPlpCovariates <- function(plp_data, minFraction = 0.1, normalize = True, removeRedundancy = True){
                  plp_data$covariateData = tidyCovariateData(plp_data$covariateData,
                                                             minFraction = minFraction,
                                                             normalize = normalize,
                                                             removeRedundancy = removeRedundancy)
                  return(plp_data)  
              }
          """)


        tidy_plp_covariates = r['tidyPlpCovariates']
        tidy_plp_data = tidy_plp_covariates(plp_data,
                                            minFraction = self._tidy_plp_settings['min_fraction'],
                                            normalize = self._tidy_plp_settings['normalize'],
                                            removeRedundancy = self._tidy_plp_settings['remove_redundancy'])
        return tidy_plp_data
        



    def __baseline_covariate_descriptions__(self, plp_data):
        """
        Gets descriptions of baseline covariates from a plpData object
        @param plp_data: An R object of type plpData, containing information on the cohorts, their outcomes, and baseline covariates
        @return: a pandas dataframe describing the covariates that have been extracted

        """
        print('Constructing baseline covariate descriptions')
        r("""
              getCovariateRefDataFrame <- function(plp_data){
                  return(data.frame(plp_data$covariateData$covariateRef))
              }
          """)

        
        get_covariate_ref_data_frame = r['getCovariateRefDataFrame']
        covariate_ref_df = get_covariate_ref_data_frame(plp_data )
        condition_prefix = 'condition_occurrence any time prior through 0 days relative to index:'
        covariate_ref_df['covariateName'] = [i.replace(condition_prefix,'') for i in covariate_ref_df['covariateName']]

        return covariate_ref_df



    def __baseline_covariate_data__(self, plp_data):
        """
        Gets baseline covariates for each subject from a plpData R object as a pandas dataframe in the wide format
        """
        print('Constructing baseline covariates')
        r("""
              getCovariateDataFrame <- function(plp_data){
                  target_cohort <- data.frame(plp_data$cohorts[,c('rowId','subjectId')])
                  covariate_ref_df <- data.frame(plp_data$covariateData$covariateRef)
                  covariates_df_long <- data.frame(plp_data$covariateData$covariates)
                  covariates_df_long <- merge(covariates_df_long, covariate_ref_df,by ='covariateId')
                  covariates_df_long <- merge(target_cohort, covariates_df_long, by='rowId')
                  return(covariates_df_long[,c('subjectId','covariateName','covariateValue')])
              }
          """)
        
        get_covariate_data_frame = r['getCovariateDataFrame']
        df_long = get_covariate_data_frame(plp_data)

        condition_prefix = 'condition_occurrence any time prior through 0 days relative to index:'
        df_long['covariateName'] = [i.replace(condition_prefix,'') for i in df_long['covariateName']]

        baseline_covariate_data = df_long.pivot_table(index= 'subjectId',  columns = 'covariateName',  values =  'covariateValue').fillna(0).reset_index()
        
        return baseline_covariate_data


    
    
       
            
    def __target_cohort_subject_ids__ (self, plp_data):
        """
        Gets target cohort subject ids
        @param plp_data: An R object of type plpData, containing information on the cohorts, their outcomes, and baseline covariates
        @return: A list of subject ids in the target cohort sample
        """
        r("""
          getTargetCohortSubjectIds <- function(plp_data){
                  return(plp_data$cohorts$subjectId)
          }
        """)
        print('Fetching list of subject ids in the target cohort sample')
        get_target_cohort_subject_ids = r['getTargetCohortSubjectIds']
        subject_ids = get_target_cohort_subject_ids(plp_data)
        subject_ids = [int(i) for i in subject_ids]
        
        return subject_ids
    


    def __r_temporal_covariate_data__(self, subject_ids):
        """
        Extracts temporal covariate data for the subjects in a target cohort using a custom covariate builder
        @param subject_ids: Ids of subjects in the sample
        return: An R object of type CovariateData, containing information on temporal covariates
        """

        r("""

            getTemporalCovariateData <- function(hostname, 
                                                 port, 
                                                 dbname, 
                                                 user,
                                                 password,
                                                 cdmDatabaseSchema,
                                                 cohortDatabaseSchema,
                                                 cohortTable,
                                                 cohortId,
                                                 subjectIds,
                                                 covariateSettings, 
                                                 path) {
              writeLines("Constructing temporal covariates")
              if (length(covariateSettings$includedCovariateConceptIds) == 0) {
                
                return(NULL)
              }

              # SQL to fetch the covariate data:
              sql <- sprintf(paste("SELECT c.subject_id as subject_id,",
                                   "v.visit_detail_id as stay_id,",
                                   "v.visit_start_datetime as stay_start,",
                                   "v.visit_end_datetime as stay_end,",
                                   "EXTRACT(EPOCH FROM (m.measurement_datetime - v.visit_start_datetime))/3600 as hours,",
                                   "m.measurement_concept_id as covariate_id,",
                                   "m.value_as_number as covariate_value",
                                   "FROM (SELECT *",
                                   "FROM %s.%s c",
                                   "WHERE c.cohort_definition_id = %s",
                                   "AND c.subject_id IN (%s)) c,",
                                   "%s.visit_detail v,",
                                   "(SELECT *",
                                   "FROM %s.measurement m",
                                   "WHERE m.person_id IN (%s)",
                                   "AND m.measurement_concept_id IN (%s)) m",
                                   "WHERE c.subject_id = v.person_id",
                                   "AND c.subject_id = m.person_id",
                                   "AND v.visit_detail_id = m.visit_detail_id",
                                   "AND m.measurement_datetime >= v.visit_start_datetime",
                                   "AND m.measurement_datetime <= v.visit_start_datetime + INTERVAL \'2 day\'"
              ), cohortDatabaseSchema,
              cohortTable,
              cohortId, 
              paste(subjectIds, collapse = ", "), 
              cdmDatabaseSchema, 
              cdmDatabaseSchema, 
              paste(subjectIds, collapse = ", "),
              paste(covariateSettings$includedCovariateConceptIds, collapse = ", "))

              path_to_covariate_data <-paste(path,'covariates.csv',sep='')
              command <- sprintf("%scopy (%s) to %s with csv header","\\\\",sql, path_to_covariate_data)

              #execute
              system(sprintf('PGPASSWORD=%s psql -h %s -p %s -d %s -U %s -c "%s"', 
                             password, 
                             hostname, 
                             port, 
                             dbname, 
                             user, 
                             command))


              # Retrieve the covariates:
              covariates <- read.csv(path_to_covariate_data)
              colnames(covariates) <- gsub('_','',gsub("(_[a-z])","\\\\U\\\\1",colnames(covariates),perl=TRUE))

              covariates$stayStart <- as.character(covariates$stayStart)

              covariates$stayEnd <- as.character(covariates$stayEnd)


              # SQL to fetch covariate reference:
              sql2 <- sprintf(paste(
                "SELECT c.concept_id as covariate_id,",
                "c.concept_name as covariate_name",
                "FROM %s.concept c",
                "WHERE c.concept_id IN (%s)"
              ),cdmDatabaseSchema,
              paste(covariateSettings$includedCovariateConceptIds, collapse = ", "))

              path_to_covariate_refs <-paste(path,'covariate_references.csv',sep='')
              command2 <- sprintf("%scopy (%s) to %s with csv header","\\\\",sql2, path_to_covariate_refs)

              #execute
              system(sprintf('PGPASSWORD=%s psql -h %s -p %s -d %s -U %s -c "%s"', 
                             password, 
                             hostname, 
                             port, 
                             dbname, 
                             user, 
                             command2))

              covariateRef <- read.csv(path_to_covariate_refs)
              colnames(covariateRef) <- gsub('_','',gsub("(_[a-z])","\\\\U\\\\1",colnames(covariateRef),perl=TRUE))

              # Construct analysis reference:
              analysisRef <- data.frame(analysisId = 1,
                                        analysisName = "Selected Temporal Covariates",
                                        domainId = "Measurement",
                                        startDay = 0,
                                        endDay = 0,
                                        isBinary = "N",
                                        missingMeansZero = "N")

              # Construct analysis reference:
              metaData <- list(sql = sql, call = match.call())

              result <- Andromeda::andromeda(covariates = covariates,
                                             covariateRef = covariateRef,
                                             analysisRef = analysisRef)
              attr(result, "metaData") <- metaData
              class(result) <- "CovariateData"
              return(result)
            }


          """)
        
        path = self._working_directory + '/data'
        if not os.path.exists(path):
            os.makedirs(path)  
        
        get_temporal_covariate_data = r['getTemporalCovariateData']
        r_temporal_covariate_data = get_temporal_covariate_data(hostname = self._db_connection_details['hostname'], 
                                                 port = self._db_connection_details['port'], 
                                                 dbname = self._db_connection_details['dbname'], 
                                                 user = self._db_connection_details['user'],
                                                 password = self._db_connection_details['password'],
                                                 cdmDatabaseSchema = self._cohort_details['cdm_database_schema'],
                                                 cohortDatabaseSchema = self._cohort_details['target_cohort_database_schema'],
                                                 cohortTable = self._cohort_details['target_cohort_table'],
                                                 cohortId = self._cohort_details['target_cohort_id'],
                                                 subjectIds = subject_ids,
                                                 covariateSettings =  self._temporal_covariate_settings, 
                                                 path = path)

        return r_temporal_covariate_data


    def __temporal_covariate_data_long__(self, r_temporal_covariate_data):
        """
        Gets temporal covariates for each subject as a pandas dataframe in the wide format
        """
        r("""
              getTemporalCovariateDataLong <- function(temporal_covariate_data){
                  temporal_covariate_ref_df <- temporal_covariate_data$covariateRef
                  temporal_covariates_df_long <- temporal_covariate_data$covariates
                  temporal_covariates_df_long <-merge(temporal_covariates_df_long,temporal_covariate_ref_df, by ='covariateId')
                  return(temporal_covariates_df_long[,c('subjectId','stayId','stayStart', 'stayEnd','hours', 'covariateId', 'covariateName','covariateValue')])
              }
          """)
        print('Fetching temporal covariates as a pandas dataframe in the long format')
        get_temporal_covariate_data_long = r['getTemporalCovariateDataLong']
        df_long = get_temporal_covariate_data_long(r_temporal_covariate_data)
        df_long['stayId'] = df_long.apply(lambda x: str(int(x.subjectId))+'_'+ str(int(x.stayId)), axis =1)
        
        return df_long
        

    def __covariate_value_mapping__(self, df):
        """
        Remaps covariate values using provided mappings 
        """
        mappings = dict()
        for key in self._expt_config_settings['categorical_covariate_concept_value_mappings'].keys():
            if key in self._temporal_covariate_names.keys():
                mappings[self._temporal_covariate_names[key]] = self._expt_config_settings['categorical_covariate_concept_value_mappings'][key]
                         
        return df.replace(mappings)
        
        
    def __temporal_covariate_data_wide__(self, df_long):
        """
        Gets temporal covariates for each subject as a pandas dataframe in the wide format
        """
        print('Constructing temporal covariates for each subject as a pandas dataframe in the wide format')
        df_wide =  df_long.pivot_table(index = ['subjectId','stayId','hours'],  columns = 'covariateName',  values =  'covariateValue').reset_index()
        df_wide.columns.name = None
        df_wide = df_wide.sort_values(by=['stayId', 'hours'],ascending=True)
        df_wide['seqnum'] = df_wide.groupby(['stayId']).cumcount()

        id_cols = ['subjectId','stayId','seqnum','hours']
        covar_cols = sorted(list(set(df_wide.columns) - set(id_cols)))
        df_wide = df_wide[id_cols + covar_cols]
        df_wide = self.__covariate_value_mapping__(df_wide)

        return df_wide


    

    def __outcomes_dict__(self, plp_data):
        """
        Gets outcomes from a plpData R object
        @param plp_data: An R object of type plpData, containing information on the cohorts, their outcomes, and baseline covariates
        @return: A python dictionary where the key is the subject id, and the value is the date of the outcome

        """
        print('Constructing outcomes dictionary')
        r("""
              getOutcomeData <- function(plp_data){
                  target_cohort <- data.frame(plp_data$cohorts[,c('rowId','subjectId','cohortStartDate')])
                  outcome_cohort <- data.frame(plp_data$outcomes[,c('rowId','daysToEvent')])
                  outcome_cohort <- merge(x= target_cohort,y =outcome_cohort,by ='rowId')
                  outcome_cohort$y_true_date <- outcome_cohort$cohortStartDate + outcome_cohort$daysToEvent
                  outcome_cohort$y_true_date<- as.character(outcome_cohort$y_true_date)
                  return(outcome_cohort[,c('subjectId','y_true_date')])
              }
          """)

        get_outcome_data = r['getOutcomeData']
        outcome_df = get_outcome_data (plp_data)
        outcome_dict = dict(zip(outcome_df['subjectId'].astype(int),pd.to_datetime(outcome_df['y_true_date'], format='%Y-%m-%d')))
        return outcome_dict



    def __outcomes_per_stay__(self, df_long, df_wide, outcome_dict):
        """
        Gets outcomes per stay for each subject
        @param df_long: A pandas dataframe with the covariate observations for each person in the long format
        @param df_wide: A pandas dataframe with the temporal covariate observations for each person in the wide format
        @param outcome_dict: A python dictionary where the key is the subject id, and the value is the date of the outcome
        @return: A pandas dataframe with outcomes per stay per subject
        """
        print('Constructing outcomes per stay per subject')
        outcome_per_stay  = df_long[['subjectId','stayId','stayStart', 'stayEnd']].drop_duplicates()
        outcome_per_stay = outcome_per_stay[outcome_per_stay['stayId'].isin(set(df_wide['stayId']))]
        outcome_per_stay['stayStart'] = pd.to_datetime(outcome_per_stay['stayStart'])
        outcome_per_stay['stayEnd'] = pd.to_datetime(outcome_per_stay['stayEnd'])
        outcome_per_stay['y_true'] = 0

        for subject in outcome_dict.keys():
            outcome_date = pd.to_datetime(outcome_dict[subject])
            outcome_per_stay.loc[((outcome_per_stay['subjectId']==subject) & 
                                  (outcome_per_stay['stayStart'] <= outcome_date) & 
                                  (outcome_per_stay['stayEnd'] >= outcome_date)), 'y_true'] = 1 
        return outcome_per_stay


    
    def __train_val_split__(self, temporal_covariate_data, outcome_per_stay, baseline_covariate_data):
        
        print('Splitting data')

        person_level_data = outcome_per_stay[['subjectId','y_true']].drop_duplicates()

        X = person_level_data['subjectId']
        y = person_level_data['y_true']

        X_train, X_val, _, _ = train_test_split( X, 
                                                y, 
                                                test_size=self._model_training_settings['val_size'], 
                                                random_state= self._model_training_settings['random_state'], 
                                                stratify=y)

        train_subjects = sorted(X_train)
        val_subjects = sorted(X_val)

        X_train_data = temporal_covariate_data[temporal_covariate_data['subjectId'].isin(train_subjects)]
        X_train_data = X_train_data.drop('subjectId',1)


        y_train_data = outcome_per_stay[outcome_per_stay['subjectId'].isin(train_subjects)]
        y_train_data = y_train_data[['stayId', 'y_true']]

        X_val_data = temporal_covariate_data[temporal_covariate_data['subjectId'].isin(val_subjects)]
        X_val_data = X_val_data.drop('subjectId',1)


        y_val_data = outcome_per_stay[outcome_per_stay['subjectId'].isin(val_subjects)]
        y_val_data = y_val_data[['stayId', 'y_true']]
        
        X_train_baseline = baseline_covariate_data[baseline_covariate_data['subjectId'].isin(train_subjects)]
        X_val_baseline = baseline_covariate_data[baseline_covariate_data['subjectId'].isin(val_subjects)]
        
        return X_train_data, X_val_data, y_train_data, y_val_data, X_train_baseline, X_val_baseline

    
    def __file_name__(self,
                      typ,
                      dset
                     ):
        filename  = '{}_T{}_O{}_{}_{}.csv'.format(self._analysis_name.upper(),  
                                                 self._cohort_details['target_cohort_id'], 
                                                 self._cohort_details['outcome_cohort_id'],
                                                 typ.upper(), 
                                                 dset.upper())
        return filename
      
    def __domains__(self, df, covariate_names, categorical_covariates):
        """
        Get domains of categorical covariates
        """
        df = df[['covariateId','covariateValue']].drop_duplicates().dropna()
        
        dictionary = dict()
        for covariate in categorical_covariates:
            covariate_name = covariate_names[covariate]
            covariate_values = list(df['covariateValue'][df['covariateId'] == covariate])
            covariate_value_mapping = self._expt_config_settings['categorical_covariate_concept_value_mappings'][covariate]
            covariate_value_names = [covariate_value_mapping[i] for i in covariate_values]
            dictionary[covariate_name] = covariate_value_names
           
        return dictionary

    
    def __yaml_doc__(self,
                     df,
                     train_tgt_file = 'data/y_train',
                     train_feat_file = 'data/X_train',
                     train_baseline_feat_file = 'data/X_train_baseline',
                     val_tgt_file = 'data/y_val',
                     val_feat_file = 'data/X_val',
                     val_baseline_feat_file = 'data/X_val_baseline',
                     tgt_col = 'y_true',
                     idx_cols = 'stayId',
                     time_order_col = ['hours', 'seqnum'],
                     feat_cols = None,
                     numerical = list(),
                     normal_values = dict()
                 ):
        
        expt_config = dict()
        expt_config['tgt_col'] = tgt_col
        expt_config['idx_cols'] = idx_cols
        expt_config['time_order_col'] = time_order_col
        
        expt_config['feat_cols'] = feat_cols
        expt_config['train'] = dict()
        expt_config['train']['tgt_file'] = train_tgt_file
        expt_config['train']['feat_file'] = train_feat_file
        expt_config['train']['baseline_feat_file'] = train_baseline_feat_file
        
        expt_config['val'] = dict()
        expt_config['val']['tgt_file'] = val_tgt_file
        expt_config['val']['feat_file'] = val_feat_file
        expt_config['val']['baseline_feat_file'] = val_baseline_feat_file
        
        categorical_covariates = sorted(list(set(self._expt_config_settings['categorical_covariate_concept_ids'] ).intersection(set(self._temporal_covariate_names.keys()))))
        
        expt_config['category_map'] = self.__domains__(df, self._temporal_covariate_names, categorical_covariates)
        
        numerical_covariates = sorted(list(set(self._expt_config_settings['numerical_covariate_concept_ids'] ).intersection(set(self._temporal_covariate_names.keys()))))
        
        expt_config['numerical'] = sorted([self._temporal_covariate_names[i] for i in numerical_covariates])
        
    
        normal_value_covariate_ids = sorted(list(set(self._expt_config_settings['normal_covariate_concept_values'].keys()).intersection(set(self._temporal_covariate_names.keys()))))
        
        normal_values = dict()
        for covariate in normal_value_covariate_ids:
            normal_values[self._temporal_covariate_names[covariate]] = self._expt_config_settings['normal_covariate_concept_values'][covariate]
        expt_config['normal_values'] = normal_values

        expt_config_filename  = '{}_T{}_O{}_expt_config.yaml'.format(self._analysis_name.upper(),         
                                                                     self._cohort_details['target_cohort_id'],
                                                                     self._cohort_details['outcome_cohort_id']
                                                                    )
        with open(expt_config_filename, 'w') as outfile:
            yaml.dump(expt_config, outfile, default_flow_style=False, sort_keys=False) 
            
        print('Experiment configurations saved to {}'.format(expt_config_filename)) 
        
       
    def __output_dir__(self):
        path = os.path.join(self._working_directory, self._output_directory)
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path)
        
        
    def __training_setup__(self):
        print('Extracting training features')
        
        plp_data = self.__get_plp_data__()
        
        tidy_plp_data = self.__tidy_plp_covariates__(plp_data)
        
        baseline_covariate_descriptions = self.__baseline_covariate_descriptions__(tidy_plp_data)
        
        baseline_covariate_data = self.__baseline_covariate_data__(tidy_plp_data)
        
        target_cohort_subject_ids = self.__target_cohort_subject_ids__(tidy_plp_data)
        
        r_temporal_covariate_data = self.__r_temporal_covariate_data__(target_cohort_subject_ids)
        
        temporal_covariate_data_long = self.__temporal_covariate_data_long__(r_temporal_covariate_data)
        
        self._temporal_covariate_names = pd.Series(temporal_covariate_data_long.covariateName.values, index=temporal_covariate_data_long.covariateId).to_dict()
        
        temporal_covariate_data_wide = self.__temporal_covariate_data_wide__(temporal_covariate_data_long)
       
        outcomes_dict = self.__outcomes_dict__(tidy_plp_data)
        
        outcomes_per_stay = self.__outcomes_per_stay__(temporal_covariate_data_long,
                                                       temporal_covariate_data_wide,
                                                       outcomes_dict
                                                      )

            
        X_train, X_val, y_train, y_val, X_train_baseline, X_val_baseline = self.__train_val_split__(temporal_covariate_data_wide, 
                                                                                                    outcomes_per_stay,
                                                                                                    baseline_covariate_data)
        
        X_train_file_name = self.__file_name__('FEAT','train')
        
        X_val_file_name = self.__file_name__('FEAT','val') 
        
        y_train_file_name = self.__file_name__('COHORT_OUT', 'train')
        
        y_val_file_name = self.__file_name__('COHORT_OUT', 'val')
        
        
        X_train_baseline_file_name = self.__file_name__('FEAT','train_baseline')
        
        X_val_baseline_file_name = self.__file_name__('FEAT','val_baseline') 
        
        out_dir = self.__output_dir__()
       
        X_train.to_csv(out_dir/X_train_file_name, index=False)
        print('Training feature data saved to {}'.format(out_dir/X_train_file_name))
        
        X_val.to_csv(out_dir/X_val_file_name, index=False)
        print('Validation feature data saved to {}'.format(out_dir/X_val_file_name))
        
        y_train.to_csv(out_dir/y_train_file_name, index=False)
        print('Training outcome data saved to {}'.format(out_dir/y_train_file_name))
        
        y_val.to_csv(out_dir/y_val_file_name, index=False)
        print('Validation outcome data saved to {}'.format(out_dir/y_val_file_name))
        
        X_train_baseline.to_csv(out_dir/X_train_baseline_file_name, index=False)
        print('Training baseline feature data saved to {}'.format(out_dir/X_train_baseline_file_name))
        
        X_val_baseline.to_csv(out_dir/X_val_baseline_file_name, index=False)
        print('Validation baseline feature data saved to {}'.format(out_dir/X_val_baseline_file_name))
        
        self.__yaml_doc__(temporal_covariate_data_long,
                          train_tgt_file = '{DATA_DIR}/' + y_train_file_name,
                          train_feat_file = '{DATA_DIR}/' + X_train_file_name,
                          train_baseline_feat_file = '{DATA_DIR}/' + X_train_baseline_file_name,
                          val_tgt_file = '{DATA_DIR}/' + y_val_file_name,
                          val_feat_file = '{DATA_DIR}/' + X_val_file_name,
                          val_baseline_feat_file = '{DATA_DIR}/' + X_val_baseline_file_name,
                          tgt_col = 'y_true',
                          idx_cols = 'stay',
                          time_order_col = ['hours', 'seqnum'],
                          feat_cols = None,
                          numerical = list(),
                          normal_values = dict()
                     )

        
        return 
    
    
    def __prediction_setup__(self, **kwargs):
        
        subject_ids = []
        
        if 'subject_id' not in kwargs.keys():
            print("'subject_id' not specified")
            return
        
        elif isinstance(kwargs['subject_id'], int) or isinstance(kwargs['subject_id'], str):
            subject_ids = [kwargs['subject_id']]
            
        elif all((isinstance(x, int) or isinstance(x, str)) for x in kwargs['subject_id']):
            subject_ids = kwargs['subject_id']
            
        else:
            print('Invalid subject_id')
            return
                
        print('Extracting prediction features')
        
        r_temporal_covariate_data = self.__r_temporal_covariate_data__(subject_ids)
        
        temporal_covariate_data_long = self.__temporal_covariate_data_long__(r_temporal_covariate_data)
        
        if temporal_covariate_data_long.empty:
            print('No records found for the subject id(s)')
            return 
        
        else:
            subjects_not_found = list(set(subject_ids) - set(temporal_covariate_data_long['subjectId']))
            if len(subjects_not_found)>0:
                print('The following subjects were not found: {}'.format(subjects_not_found))
            if not hasattr(self, '_temporal_covariate_names'):
                self._temporal_covariate_names = pd.Series(temporal_covariate_data_long.covariateName.values, index=temporal_covariate_data_long.covariateId).to_dict()
            temporal_covariate_data_wide = self.__temporal_covariate_data_wide__(temporal_covariate_data_long)
            return temporal_covariate_data_wide
        
        
    def extract_features(self, **kwargs):
        
        if kwargs['setup'] == 'train':
            return self.__training_setup__()
        
        elif kwargs['setup'] == 'prediction':
            return self.__prediction_setup__(**kwargs)  
        
        else:
            print("""Invalid setup, permitted options are as follows:
            'training': extract data for training a model
            'prediction': extract data for prediction
                  """)
        
        