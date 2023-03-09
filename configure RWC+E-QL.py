# Utils
# from envs import env
import numpy as np
import configparser

# SQL
import sqlalchemy as db
import json


#Credenciales
config = configparser.ConfigParser()
config.read('db_config.ini')
host = config['postgres']['host']
db_name = config['postgres']['db_name']
port = config['postgres']['port']
user = config['postgres']['user']
pwd = config['postgres']['pass']


# Conexión a la DB de resultados

engine = db.create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db_name}')
metadata = db.MetaData()


try: 
    connection = engine.connect()

except db.exc.SQLAlchemyError as e:
    exit(str(e.__dict__['orig']))



datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
insertDatosEjecucion = datosEjecucion.insert().returning(datosEjecucion.c.id)


# algorithms = ['GWO_SCP_BCL','WOA_SCP_BCL','GWO_SCP_MIR','WOA_SCP_MIR','GWO_SCP_QL1','WOA_SCP_QL1','GWO_SCP_QL2','WOA_SCP_QL2','GWO_SCP_QL3','WOA_SCP_QL3','GWO_SCP_QL4','WOA_SCP_QL4','GWO_SCP_QL5','WOA_SCP_QL5','SCA_SCP_BCL','SCA_SCP_MIR','SCA_SCP_QL1','SCA_SCP_QL2','SCA_SCP_QL3','SCA_SCP_QL4','SCA_SCP_QL5']
algorithms = [
"GWO_RWC+E_QL1","GWO_RWC+E_QL2","GWO_RWC+E_QL3","GWO_RWC+E_QL4","GWO_RWC+E_QL5",
"WOA_RWC+E_QL1","WOA_RWC+E_QL2","WOA_RWC+E_QL3","WOA_RWC+E_QL4","WOA_RWC+E_QL5",
"SCA_RWC+E_QL1","SCA_RWC+E_QL2","SCA_RWC+E_QL3","SCA_RWC+E_QL4","SCA_RWC+E_QL5"
]

# algorithms = [
# 'GWO_RWE_QL1','GWO_RWE_QL2','GWO_RWE_QL3','GWO_RWE_QL4','GWO_RWE_QL5',
# 'GWO_RWE_SA1','GWO_RWE_SA2','GWO_RWE_SA3','GWO_RWE_SA4','GWO_RWE_SA5',
# 'WOA_RWE_QL1','WOA_RWE_QL2','WOA_RWE_QL3','WOA_RWE_QL4','WOA_RWE_QL5',
# 'WOA_RWE_SA1','WOA_RWE_SA2','WOA_RWE_SA3','WOA_RWE_SA4','WOA_RWE_SA5',
# 'SCA_RWE_QL1','SCA_RWE_QL2','SCA_RWE_QL3','SCA_RWE_QL4','SCA_RWE_QL5',
# 'SCA_RWE_SA1','SCA_RWE_SA2','SCA_RWE_SA3','SCA_RWE_SA4','SCA_RWE_SA5',
# 'HHO_RWE_QL1','HHO_RWE_QL2','HHO_RWE_QL3','HHO_RWE_QL4','HHO_RWE_QL5',
# 'HHO_RWE_SA1','HHO_RWE_SA2','HHO_RWE_SA3','HHO_RWE_SA4','HHO_RWE_SA5'
# ]

 
instances = ['RW300','RW350','RW400','RW450','RW500','RW550','RW600','RW650','RW700','RW750','RW800']

runs = 31
population  = 40
maxIter = 5000
beta_Dis = 0.8 #Parámetro de la discretización de RW
ql_alpha = 0.1
ql_gamma =  0.4
policy = "softMax-rulette-elitist" #puede ser 'e-greedy', 'greedy', 'e-soft', 'softMax-rulette', 'softMax-rulette-elitist'
rewardType = "withPenalty1" #pueden ser 'withoutPenalty1': osea se recompensa con +1, 'withPenalty1': osea +1 o -1
rewardTypes = ["withPenalty1", "withoutPenalty1", "globalBest", "rootAdaptation", "escalatingMultiplicativeAdaptation"]
qlAlphaType = "static" # Puede ser 'static', 'iteration', 'visits'
repair = 2 # 1:Simple; 2:Compleja
instance_dir = "MSCP/"
DS_actions = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
paramsML = {'cond_backward': '10', 'MinMax': 'min', 'DS_actions': DS_actions} 


for run in range(runs):
    for instance in instances:
        for algorithm in algorithms:
            FO = algorithm.split("_")[1].replace("RW","")
            MH = algorithm.split("_")[0]
            ML = algorithm.split("_")[2][:2]
            if ML == "QL" or ML == "SA":
                discretizationScheme = DS_actions[np.random.randint(low=0, high=len(DS_actions))]
                if algorithm.split("_")[2][2] == 1:
                    rewardType = rewardTypes[int(algorithm.split("_")[2][2]) - 1] 

            if algorithm.split("_")[2] == "V1S":
                discretizationScheme = 'V1,Standard'
            if algorithm.split("_")[2] == "V2S":
                discretizationScheme = 'V2,Standard'
            if algorithm.split("_")[2] == "V3S":
                discretizationScheme = 'V3,Standard'
            if algorithm.split("_")[2] == "V4S":
                discretizationScheme = 'V4,Standard'
            if algorithm.split("_")[2] == "S1S":
                discretizationScheme = 'S1,Standard'
            if algorithm.split("_")[2] == "S2S":
                discretizationScheme = 'S2,Standard'
            if algorithm.split("_")[2] == "S3S":
                discretizationScheme = 'S3,Standard'
            if algorithm.split("_")[2] == "S4S":
                discretizationScheme = 'S4,Standard'
            if algorithm.split("_")[2] == "V1C":
                discretizationScheme = 'V1,Complement'
            if algorithm.split("_")[2] == "V2C":
                discretizationScheme = 'V2,Complement'
            if algorithm.split("_")[2] == "V3C":
                discretizationScheme = 'V3,Complement'
            if algorithm.split("_")[2] == "V4C" or "MIR":
                discretizationScheme = 'V4,Complement'
            if algorithm.split("_")[2] == "S1C":
                discretizationScheme = 'S1,Complement'
            if algorithm.split("_")[2] == "S2C":
                discretizationScheme = 'S2,Complement'
            if algorithm.split("_")[2] == "S3C":
                discretizationScheme = 'S3,Complement'
            if algorithm.split("_")[2] == "S4C":
                discretizationScheme = 'S4,Complement'
            if algorithm.split("_")[2] == "V1E":
                discretizationScheme = 'V1,Elitist'
            if algorithm.split("_")[2] == "V2E":
                discretizationScheme = 'V2,Elitist'
            if algorithm.split("_")[2] == "V3E":
                discretizationScheme = 'V3,Elitist'
            if algorithm.split("_")[2] == "V4E" or "BCL":
                discretizationScheme = 'V4,Elitist'
            if algorithm.split("_")[2] == "S1E":
                discretizationScheme = 'S1,Elitist'
            if algorithm.split("_")[2] == "S2E":
                discretizationScheme = 'S2,Elitist'
            if algorithm.split("_")[2] == "S3E":
                discretizationScheme = 'S3,Elitist'
            if algorithm.split("_")[2] == "S4E":
                discretizationScheme = 'S4,Elitist'
            if algorithm.split("_")[2] == "V1A":
                discretizationScheme = 'V1,Static'
            if algorithm.split("_")[2] == "V2A":
                discretizationScheme = 'V2,Static'
            if algorithm.split("_")[2] == "V3A":
                discretizationScheme = 'V3,Static'
            if algorithm.split("_")[2] == "V4A":
                discretizationScheme = 'V4,Static'
            if algorithm.split("_")[2] == "S1A":
                discretizationScheme = 'S1,Static'
            if algorithm.split("_")[2] == "S2A":
                discretizationScheme = 'S2,Static'
            if algorithm.split("_")[2] == "S3A":
                discretizationScheme = 'S3,Static'
            if algorithm.split("_")[2] == "S4A":
                discretizationScheme = 'S4,Static'      
            if algorithm.split("_")[2] == "V1R":
                discretizationScheme = 'V1,ElitistRoulette'
            if algorithm.split("_")[2] == "V2R":
                discretizationScheme = 'V2,ElitistRoulette'
            if algorithm.split("_")[2] == "V3R":
                discretizationScheme = 'V3,ElitistRoulette'
            if algorithm.split("_")[2] == "V4R":
                discretizationScheme = 'V4,ElitistRoulette'
            if algorithm.split("_")[2] == "S1R":
                discretizationScheme = 'S1,ElitistRoulette'
            if algorithm.split("_")[2] == "S2R":
                discretizationScheme = 'S2,ElitistRoulette'
            if algorithm.split("_")[2] == "S3R":
                discretizationScheme = 'S3,ElitistRoulette'
            if algorithm.split("_")[2] == "S4R":
                discretizationScheme = 'S4,ElitistRoulette'                          
            else:
                discretizationScheme = algorithm.split("_")[2]
            data = {
                'nombre_algoritmo' : algorithm,
                'parametros': json.dumps({
                    'instance_name' : instance,
                    'instance_file': instance+'.txt',
                    'instance_dir': instance_dir,
                    'population': population,
                    'maxIter':maxIter,
                    'discretizationScheme':discretizationScheme,
                    'ql_alpha': ql_gamma,
                    'ql_gamma': ql_gamma,
                    'repair': repair,
                    'policy': policy,
                    'rewardType': rewardType,
                    'qlAlphaType': qlAlphaType,
                    'beta_Dis': beta_Dis,
                    'FO': FO,
                    'MH': MH,
                    'ML': ML,
                    'paramsML': paramsML
            }),
                'estado' : 'pendiente'
            }
            result = connection.execute(insertDatosEjecucion,data)
            idEjecucion = result.fetchone()[0]
            print(f'Poblado ID #:{idEjecucion}')

print("Todo poblado")