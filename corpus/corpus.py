class CorpusElement(object):
    """Class representing a single element of a corpus."""

    def __init__(self, scenario_id, scenario, fitness, replay_list, period_conflicts, saved_c_npcs, potential_conflicts, saved_p_npcs):
        
        self.scenario = scenario # input data
        self.fitness = fitness # simulator output - fitness score or others
        self.replay_list = replay_list
        self.period_conflicts = period_conflicts
        self.saved_c_npcs = saved_c_npcs
        self.potential_conflicts = potential_conflicts
        self.saved_p_npcs = saved_p_npcs
        self.parent = None
        self.scenario_id = scenario_id

    def set_parent(self, parent):
        self.parent = parent

    def oldest_ancestor(self):
        """Returns the least recently created ancestor of this corpus item."""
        current_element = self
        generations = 0
        while current_element.parent is not None:
            current_element = current_element.parent
            generations += 1
        return current_element, generations