class GridSearch():
    def __init__(self, hyperparameter_groups):
        self._hyperparameter_groups = hyperparameter_groups

    def hyperparameter_combos(self):
        return self._hyperparameter_combos(self._hyperparameter_groups, None)

    def _hyperparameter_combos(self, groups_to_add, result):
        if groups_to_add == []:
            return result

        group_to_add = groups_to_add[0]
        remaining_groups_to_add = groups_to_add[1:]

        group_to_add_maps = self._group_to_maps(group_to_add)

        if result == None:
            return self._hyperparameter_combos(remaining_groups_to_add, group_to_add_maps)
        else:
            new_result = []
            for existing_map in result:
                for new_map in group_to_add_maps:
                    new_result.append(existing_map | new_map)
            return self._hyperparameter_combos(remaining_groups_to_add, new_result)

    def _group_to_maps(self, group):
        group_keys = group[0]
        group_values = group[1:]
       
        result = []
        for value_list in group_values:
            new_map = {}
            for (key_index, key) in enumerate(group_keys):
                new_map[key] = value_list[key_index]
            result.append(new_map)

        return result
