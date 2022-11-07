import collections

RABBIT_VALUE = 5
STAG_VALUE = 20

class InteractionManager:
    def __init__(self, c_reg=None):
        self.c_reg = c_reg

    def set_reg(self, reg):
        self.c_reg = reg

    def get_interactions(self):
        # Returns a dict of characters that occupy same space
        spaces = {}
        groups = {}
        for c_key in self.c_reg.keys():
            pos = tuple(self.c_reg[c_key]["position"])
            if pos in spaces.keys():
                spaces[pos].append(c_key)
            else:
                spaces[pos] = [c_key]

        for s_key in spaces.keys():
            if len(spaces[s_key]) > 1:
                groups[s_key] = spaces[s_key]

        return groups

    def get_type_counts(self, group):
        types = str(group).replace('123\',[]', '')
        counter = collections.Counter(types)
        count = [counter['r'], counter['s'], counter['h']]
        return count

    def get_multi_type_counts(self, groups):
        counts = []
        for group_id in groups.keys():
            group = groups[group_id]
            count = self.get_type_counts(group)
            counts.append(count)
        return counts

    def get_type_from_group(self, type, group):
        members = []
        for member in group:
            if member[0] == type:
                members.append(member)
        members.sort()
        return members

    def calculate_reward(self, group):
        points = {}
        count = self.get_type_counts(group)
        r, s, h = count
        r_points = r * RABBIT_VALUE
        s_points = s * STAG_VALUE

        rabbits = self.get_type_from_group("r", group)
        stags = self.get_type_from_group("s", group)
        hunters = self.get_type_from_group("h", group)

        i, j, k = [0, 0, 0]

        while i < len(rabbits) or j < len(stags) or k < len(hunters):
            # 2 hunters + 1 stag
            if len(hunters[k:]) >= 2 and s > 0 and s_points > 0:
                points[hunters[k]] = STAG_VALUE / 2
                points[hunters[k + 1]] = STAG_VALUE / 2
                s_points -= STAG_VALUE
                points[stags[j]] = 0
                j += 1
                k += 2
            # 1 hunter + 1 rabbit
            elif len(hunters[k:]) >= 1 and r > 0 and r_points > 0:
                points[hunters[k]] = RABBIT_VALUE
                r_points -= RABBIT_VALUE
                points[rabbits[i]] = 0
                i += 1
                k += 1
            # Assign remaining characters original value
            else:
                if i < len(rabbits):
                    points[rabbits[i]] = RABBIT_VALUE
                    i += 1
                if j < len(stags):
                    points[stags[j]] = STAG_VALUE
                    j += 1
                if k < len(hunters):
                    points[hunters[k]] = 0
                    k += 1
        return points
