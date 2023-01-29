''' Configures Metric Outputs '''

train_agent_config = {
    "saveTable": False,
    "tableId" : None,
    "showMetrics" : False,
    "saveIntermMetrics" : False,
    "formatDisplay" : True
}

test_agent_config = {
    "showMap" : True,
    "saveFrames" : False,
    "formatDisplay" : True
}

test_saved_table = {
    "showMap" : True,
    "saveFrames" : False,
    "tableId" : None,
    "formatDisplay" : True
}

train_and_test_agent_config = {
    "showMetrics" : True,
    "saveMetrics" : True,
    "showTestMap" : True,
    "showIntermMetrics" : False,
    "saveIntermMetrics" : True,
    "saveFrames" : True,
    "formatDisplay" : True
}

default_metrics_config = {
    'saveMetrics' : False,
    'showMetrics' : True,
    'showTestMap' : True,
    'showIntermMetrics' : True,
    'saveIntermMetrics' : False,
    "formatDisplay" : True
}

grid_search_metrics_config = {
    'saveMetrics' : False,
    'showMetrics' : False,
    'showTestMap' : False,
    'showIntermMetrics' : False,
    'saveIntermMetrics' : False,
    "formatDisplay" : False,
    "multipleProgressBars" : True
}

def validate_config(config, valid_config):
    if config != valid_config:
        for var in valid_config.keys():
            if var not in config.keys():
                config[var] = valid_config[var]
    return config
