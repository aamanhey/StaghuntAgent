class Registry:
    def __init__(self, characters):
        '''
        - characters: A dictionary of an agent and an xy-position associated with an id
        '''
        self.characters = characters

    def add_character(self, id, character):
        if "agent" not in character.keys():
            character["agent"] = None
        if "position" not in character.keys():
            character["position"] = (-1, -1)

        self.characters[id] = character.copy()

    def remove_character(self, id):
        del self.characters[id]

    def update_character(self, id, data, item="position"):
        self.characters[id][item] = data

    def update_positions(self, new_positions):
        for c_key in new_positions:
            if c_key in self.characters.keys():
                update_character(c_key, new_positions[c_key])
            else:
                print("E: Ignoring character {} b/c not defined in character registry.".format(c_key))

    def get_ids(self):
        return self.characters.keys()

    def get_characters(self):
        return self.characters.copy()

    def get_character(self, id):
        if id in self.characters.keys():
            return self.characters[id].copy()
        else:
            print("E: No character with id {} found.".format(id))

    def get_position(self, id):
        character = self.get_character(id)
        return character["position"]

    def get_positions(self):
        positions = {}
        for c_key in self.characters.keys():
            positions[c_key] = self.characters[c_key]["position"]
        return positions

    def get_agent(self, id):
        character = self.get_character(id)
        return character["agent"]

    def get_agents(self):
        agents = {}
        for c_key in self.characters.keys():
            agents[c_key] = self.characters[c_key]["agent"]
        return agents

    def in_reg(self, characters):
        for c in characters:
            if c not in self.characters:
                print("E: {} found in map but not in character registry.".format(c))
                return False
        return True

    def reset_rewards(self):
        for id in self.characters.keys():
            agent = self.characters[id]["agent"]
            agent.reset()
