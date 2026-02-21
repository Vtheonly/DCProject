class WindSchema:
    TIMESTAMP = "timestamp"
    DATASET_SOURCE = "dataset_source"
    IS_REAL = "is_real" # 1.0 for Turkey, 0.0 for NREL Simulation
    
    WIND_SPEED = "wind_speed_ms"
    WIND_DIRECTION = "wind_direction_deg"
    ACTIVE_POWER = "active_power_kw"
    
    # Engineered Features
    WIND_CUBED = "wind_speed_cubed"
    WIND_U = "wind_u"
    WIND_V = "wind_v"
    
    FEATURES = [WIND_SPEED, WIND_DIRECTION, WIND_CUBED, WIND_U, WIND_V, IS_REAL]