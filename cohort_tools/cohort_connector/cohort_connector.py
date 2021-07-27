

import json 
from rpy2.robjects import r
from rpy2.robjects.packages import importr

base = importr('base')
database_connector = importr('DatabaseConnector')


class CohortConnector():
    """
    Class for connecting to cohort data in the OMOP Common Data Model
    """
    
    def __init__(self,
                 **kwargs
                 ):
        
        if len(kwargs.keys()) < 1:
            print("""Declare or specify path to json file with database and cohort connection parameters""")
        else:
            settings = self.__load_settings__(**kwargs)
            self.__db_connection_details__(settings, **kwargs)
            self.__cohort_details__(settings, **kwargs)
                
    
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
                settings['cohort_details']['oracle_temp_schema'] = settings['cohort_details']['oracle_temp_schema'] if settings['cohort_details']['oracle_temp_schema'] else r('NULL')
                
            return settings

                 
  
            
    def __db_connection_details__(self, settings, **kwargs):
        try:
            print('Setting connection details')
            if settings:
                self._db_connection_details = settings['connection_details']
            else:
                self._db_connection_details = {  
                                    'dbms': kwargs['dbms'],
                                    'path_to_driver': kwargs['path_to_driver'],
                                    'hostname': kwargs['hostname'],
                                    'port': kwargs['port'],
                                    'dbname': kwargs['dbname'],
                                    'user': kwargs['user'],
                                    'password': kwargs['password'],
                                  }
            print("""Successfully set connection details""")
        except:
            print("""Missing/incorrect connection details. Specify connection detail parameters as follows:
                      dbms: The type of DBMS specified as follow:
                            'oracle' for Oracle
                            'postgresql' for PostgreSQL
                            'redshift' for Amazon Redshift
                            'sql server' for Microsoft SQL Server
                            'pdw' for Microsoft Parallel Data Warehouse (PDW)
                            'netezza' for IBM Netezza
                            'bigquery' for Google BigQuery
                            'sqlite' for SQLite
                      path_to_driver: Path to the JDBC driver JAR files
                      hostname: The host name of the machine on which the server is running
                      dbname: The name of the database to connect to
                      port: (optional) The port on the server to connect to
                      user: The user name used to access the server
                      password: The password for that user
    
                    """)
            
        try:
            print("""Creating an R list of database connection details""")
            self._r_db_connection_details = database_connector.createConnectionDetails(
                                dbms = self._db_connection_details['dbms'], 
                                pathToDriver = self._db_connection_details['path_to_driver'], 
                                server = self._db_connection_details['hostname'] + '/' + self._db_connection_details['dbname'], 
                                port = self._db_connection_details['port'],
                                user = self._db_connection_details['user'], 
                                password = self._db_connection_details['password'] 
                                )
            print("""Successfully created an R list of database connection details""")
        except:
            print(""" Error creating R database connection details""")
    
    @property
    def db_connection_details(self):
        return self._db_connection_details
    
    @property
    def r_db_connection_details(self):
        return self._r_db_connection_details
    
    
            
    def __cohort_details__(self, settings, **kwargs):
        try:
            print('Setting cohort details')
            
            if settings:
                self._cohort_details = settings['cohort_details']
            else: 
                oracle_temp_schema = r('NULL')
                if 'oracle_temp_schema' in kwargs.keys():
                    oracle_temp_schema = kwargs['oracle_temp_schema'] if kwargs['oracle_temp_schema'] else r('NULL')
                print("""Creating cohort detail""")       
                self._cohort_details = {
                            'cdm_database_schema' : kwargs['cdm_database_schema'],
                            'target_cohort_database_schema' : kwargs['target_cohort_database_schema'],
                            'target_cohort_table' : kwargs['target_cohort_table'],
                            'target_cohort_id' : kwargs['target_cohort_id'],
                            'outcome_cohort_database_schema' : kwargs['outcome_cohort_database_schema'],
                            'outcome_cohort_table' : kwargs['outcome_cohort_table'],
                            'outcome_cohort_id': kwargs['outcome_cohort_id'],
                            'oracle_temp_schema' : oracle_temp_schema
                           }
            print("""Successfully set cohort details""")
        except:
            print("""Missing/incorrect cohort details. Specify cohort detail parameters as follows:
                     cdm_database_schema: The name of the database schema that contains the OMOP CDM instance
                     target_cohort_database_schema: The name of the database schema that contains the at-risk (target) cohort 
                     target_cohort_table: The name of the table that contains the at-risk (target) cohort
                     target_cohort_id: Unique Id of the at-risk (target) cohort
                     outcome_cohort_database_schema: The name of the database schema that contains the outcome cohort 
                     outcome_cohort_table: The name of the table that contains the outcome cohort
                     outcome_cohort_id: Unique Id of the the outcome cohort 
                     oracle_temp_schema: the name of the oracle database schema where temporary tables will be managed (for oracle only)
            
                 """)
            
        
    @property
    def cohort_details(self):
        return self._cohort_details
    
    

        
        