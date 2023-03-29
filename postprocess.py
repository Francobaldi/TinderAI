import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def ape(y_true, y_pred):
    if y_true == 0:
        return abs(y_pred)
    else:
        return abs(y_pred - y_true) / abs(y_true)

targets = ['INFERENCE_TIME', 
           'POWER_CONSUMPTION',
           'CPU_LOAD', 'GPU_LOAD',  
           'CPU_MEM', 'GPU_MEM',  
           'CPU_MEM_PEAK', 'GPU_MEM_PEAK',
           ]

####################################################################################################
# General Evaluation
####################################################################################################
general_evaluation = pd.read_csv('results/general-evaluation/predictions_INFERENCE_TIME.csv')
general_evaluation = pd.DataFrame({'MODEL' : [col.split('_')[0] for col in general_evaluation.columns if col not in ['True', 'Set']]})
general_evaluation = pd.DataFrame({'MODEL' : general_evaluation['MODEL'].unique()})

for target in targets:
    predictions = pd.read_csv(f"results/general-evaluation/predictions_{target}.csv")
    performance = {'MODEL' : [], 'Seed' : [], f'{target}' : []}

    for col in [col for col in predictions.columns if col not in ['True', 'Set']]:
        predictions[f'{col}_ape'] = predictions.apply(lambda row : ape(y_true = row['True'], y_pred = row[col]), axis = 1)
        performance['MODEL'].append(col.split('_')[0])
        performance['Seed'].append(col.split('_')[1])
        performance[f'{target}'].append(predictions.loc[predictions['Set'] == 'Test', f'{col}_ape'].mean())
    performance = pd.DataFrame(performance)
    performance = performance.groupby('MODEL').mean()
    performance['MODEL'] = performance.index
    performance = performance.reset_index(drop = True)
    general_evaluation = pd.merge(general_evaluation, performance, on = ['MODEL'])

general_evaluation = general_evaluation.round({col : 2 for col in general_evaluation.columns if col != 'MODEL'})
general_evaluation.to_csv('results/general-evaluation/general-evaluation.tex', sep = '&', index = False)
selected_models = ['DT20', 'DT25', 'DT30', 'RF50', 'RF100', 'RF150']

####################################################################################################
# Task Reduction
####################################################################################################
palette = sns.color_palette()
palette = palette.as_hex()
palette = {
        'task-generic' : palette[0],
        'reduced-task-generic' : palette[1],
        'face-landmarks-detection' : palette[2],
        'bodypose-estimation' : palette[3]
        }

for target in targets:
    red_tasks = pd.read_csv(f"results/task-reduction/predictions_{target}_100.csv")
    be = pd.read_csv(f"results/task-reduction/predictions_{target}_bodypose-estimation.csv")
    fld = pd.read_csv(f"results/task-reduction/predictions_{target}_face-landmarks-detection.csv")

    performance = {'Model' : [], 'Seed' : [], 
                   'reduced-task-generic' : [], 'bodypose-estimation' : [], 'face-landmarks-detection' : [],
                   }
    
    for col in [col for col in red_tasks.columns if col not in ['True', 'Set']]:
        red_tasks[f'{col}_ape'] = red_tasks.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        be[f'{col}_ape'] = be.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        fld[f'{col}_ape'] = fld.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
   
        performance['Model'].append(col.split('_')[0])
        performance['Seed'].append(col.split('_')[1])
        performance['reduced-task-generic'].append(red_tasks.loc[red_tasks['Set'] == 'Test', f'{col}_ape'].mean())
        performance['bodypose-estimation'].append(be.loc[be['Set'] == 'Test', f'{col}_ape'].mean())
        performance['face-landmarks-detection'].append(fld.loc[fld['Set'] == 'Test', f'{col}_ape'].mean())
    performance = pd.DataFrame(performance)
    performance = performance.groupby('Model').mean()
    performance['MODEL'] = performance.index
    performance = performance.reset_index(drop = True)
    performance = pd.merge(general_evaluation[['MODEL', target]], performance, on = ['MODEL'])
    performance = performance.rename(columns = {target : 'task-generic'})

    performance = performance[performance['MODEL'].isin(selected_models)]
    performance['MODEL'] = pd.Categorical(performance['MODEL'],categories = selected_models)
    performance = performance.sort_values(by = 'MODEL')
    
    plot = performance.plot.bar(x ='MODEL', y = ['task-generic', 'reduced-task-generic', 'face-landmarks-detection', 'bodypose-estimation'], 
                rot = 0, figsize = (18,12),
                xlabel = '', ylabel = '', fontsize = 35,
                yticks = [0.1, 0.2, 0.3, 0.4, 0.5],
                color = [palette[version] for version in ['task-generic', 'reduced-task-generic', 'face-landmarks-detection','bodypose-estimation']]
                )
    plt.legend(prop={'size':35}, loc = 'upper left')
    #plt.show()
    plot.figure.savefig(f"results/task-reduction/plots/{target}.pdf", bbox_inches='tight')

####################################################################################################
# Platform Reduction
####################################################################################################
palette = sns.color_palette()
palette = palette.as_hex()
palette = {
        'platform-generic' : palette[0],
        'cpu-platform-generic' : palette[4],
        'reduced-platform-generic' : palette[1],
        'jetson-nano2' : palette[2],
        'jetson-xav-agx' : palette[3]
        }

################## CPU processors ##################
cpu_only_targets = ['CPU_LOAD','CPU_MEM','CPU_MEM_PEAK','INFERENCE_TIME']
for target in [target for target in cpu_only_targets if target in targets]:
    cpu_platforms = pd.read_csv(f"results/platform-reduction/predictions_{target}_cpu-only.csv")
    red_platforms = pd.read_csv(f"results/platform-reduction/predictions_{target}_300.csv")
    nano = pd.read_csv(f"results/platform-reduction/predictions_{target}_jetson-nano2.csv")
    agx = pd.read_csv(f"results/platform-reduction/predictions_{target}_jetson-xav-agx.csv")
    
    performance = {'MODEL' : [], 'Seed' : [], 
            "cpu-platform-generic" : [], 'reduced-platform-generic' : [], 'jetson-nano2' : [], 'jetson-xav-agx' : []
                }
    
    for col in [col for col in red_platforms.columns if col not in ['True', 'Set']]:
        cpu_platforms[f'{col}_ape'] = cpu_platforms.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        red_platforms[f'{col}_ape'] = red_platforms.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        nano[f'{col}_ape'] = nano.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        agx[f'{col}_ape'] = agx.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
   
        performance['MODEL'].append(col.split('_')[0])
        performance['Seed'].append(col.split('_')[1])
        performance['cpu-platform-generic'].append(cpu_platforms.loc[cpu_platforms['Set'] == 'Test', f'{col}_ape'].mean())
        performance['reduced-platform-generic'].append(red_platforms.loc[red_platforms['Set'] == 'Test', f'{col}_ape'].mean())
        performance['jetson-nano2'].append(nano.loc[nano['Set'] == 'Test', f'{col}_ape'].mean())
        performance['jetson-xav-agx'].append(agx.loc[agx['Set'] == 'Test', f'{col}_ape'].mean())
    performance = pd.DataFrame(performance)
    performance = performance.groupby('MODEL').mean()
    performance['MODEL'] = performance.index
    performance = performance.reset_index(drop = True)
    performance = pd.merge(general_evaluation[['MODEL', target]], performance, on = ['MODEL'])
    performance = performance.rename(columns = {target : 'platform-generic'})

    performance = performance[performance['MODEL'].isin(selected_models)]
    performance['MODEL'] = pd.Categorical(performance['MODEL'],categories = selected_models)
    performance = performance.sort_values(by = 'MODEL')
    
    plot = performance.plot.bar(x ='MODEL', y = ['platform-generic', 'cpu-platform-generic', 'reduced-platform-generic', 'jetson-nano2', 'jetson-xav-agx'],
                rot = 0, figsize = (18,12),
                xlabel = '', ylabel = '', fontsize = 35,
                yticks = [0.1, 0.2, 0.3, 0.4, 0.5],
                color = [palette[version] for version in ['platform-generic', 'cpu-platform-generic', 'reduced-platform-generic', 'jetson-nano2', 'jetson-xav-agx']]
                )
    plt.legend(prop={'size':35}, loc = 'upper left')
    #plt.show()
    plot.figure.savefig(f"results/platform-reduction/plots/{target}.pdf", bbox_inches='tight')

################## All processors ##################
for target in [target for target in targets if target not in cpu_only_targets]:
    red_platforms = pd.read_csv(f"results/platform-reduction/predictions_{target}_300.csv")
    nano = pd.read_csv(f"results/platform-reduction/predictions_{target}_jetson-nano2.csv")
    agx = pd.read_csv(f"results/platform-reduction/predictions_{target}_jetson-xav-agx.csv")
    
    performance = {'MODEL' : [], 'Seed' : [], 
            'reduced-platform-generic' : [], 'jetson-nano2' : [], 'jetson-xav-agx' : []}
    
    for col in [col for col in red_platforms.columns if col not in ['True', 'Set']]:
        red_platforms[f'{col}_ape'] = red_platforms.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        nano[f'{col}_ape'] = nano.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
        agx[f'{col}_ape'] = agx.apply(lambda row : ape(y_pred = row[col], y_true = row['True']), axis = 1)
   
        performance['MODEL'].append(col.split('_')[0])
        performance['Seed'].append(col.split('_')[1])
        performance['reduced-platform-generic'].append(red_platforms.loc[red_platforms['Set'] == 'Test', f'{col}_ape'].mean())
        performance['jetson-nano2'].append(nano.loc[nano['Set'] == 'Test', f'{col}_ape'].mean())
        performance['jetson-xav-agx'].append(agx.loc[agx['Set'] == 'Test', f'{col}_ape'].mean())
    performance = pd.DataFrame(performance)
    performance = performance.groupby('MODEL').mean()
    performance['MODEL'] = performance.index
    performance = performance.reset_index(drop = True)
    performance = pd.merge(general_evaluation[['MODEL', target]], performance, on = ['MODEL'])
    performance = performance.rename(columns = {target : 'platform-generic'})

    performance = performance[performance['MODEL'].isin(selected_models)]
    performance['MODEL'] = pd.Categorical(performance['MODEL'],categories = selected_models)
    performance = performance.sort_values(by = 'MODEL')
    
    plot = performance.plot.bar(x ='MODEL', y = ['platform-generic', 'reduced-platform-generic', 'jetson-nano2', 'jetson-xav-agx'],
                rot = 0, figsize = (18,12),
                xlabel = '', ylabel = '', fontsize = 35,
                yticks = [0.1, 0.2, 0.3, 0.4, 0.5],
                color = [palette[version] for version in ['platform-generic', 'reduced-platform-generic', 'jetson-nano2', 'jetson-xav-agx']]
                )
    plt.legend(prop={'size':35}, loc = 'upper left')
    #plt.show()
    plot.figure.savefig(f"results/platform-reduction/plots/{target}.pdf", bbox_inches='tight')

