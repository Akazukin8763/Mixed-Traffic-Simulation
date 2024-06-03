from enum import Enum


class RoadUser(Enum):
    Pedestrian = 'Pedestrian'
    Vehicle = 'Vehicle'
    Bicycle = 'Bicycle'

    def __str__(self):
        return str(self.value)
        
class UserID:
    def __init__(self, user_type: RoadUser, user_ID: int):
        self.user_type = user_type
        self.user_ID = user_ID
