
import numpy

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random


PLAYER_SELF = 1
N_MARINES = 5
CENTRE_DE_COMMANDEMENT = 18
CASERNES = 21
BUNKER = 24
MARINES = 48

UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
coord_x = features.FeatureUnit.x
coord_y = features.FeatureUnit.y

_NOT_QUEUED = [0]
_SCREEN = [0]
_SELECT_ALL = [0]


class ProjetIA(base_agent.BaseAgent):
    def reset(self):
        self._NONE = 0
        self.SCV_INACTIF_SELECTIONNE = 1
        self.RANDOM_SCV_SELECTIONNE = 2
        self.CASERNES_SELECTIONNE = 3
        self.CDC_SELECTIONNE = 4
        self.RANDOM_MARINE_SELECTIONNE = 5
        self.BUNKER_SELECTIONNE = 6
        self.ARMEE_SELECTIONNE = 7
        self.lst = [False, False, False, False, False, False, False, False]
        self.lst_elem = self.lst[:]


        self.nombre_SCV = 12
        self.nombre_scv_CEC = 0

        self.nombres_marines = 0
        self.nombre_raffinerie = 0
        self.nombre_caserne = 0
        self.nombre_bunker = 0
        self.nombre_ravitaillement = 0
        self.nombre_centreTechnique = 0

        self.nombre_marines_CEC = 0
        self.nombre_raffinerie_CEC = 0
        self.nombre_ravitaillement_CEC = 0
        self.nombre_centreTechnique_CEC = 0
        self.nombre_caserne_CEC = 0
        self.nombre_bunker_CEC = 0

        self.mode_attaque = 0
        self.structures = 400
        self.mode_patrouille = 0
        self.premier_scv = False
        self.phase_offensive = False

    def is_elem_selected(self, x):
        return self.lst_elem[x]

    def select_element(self, x):
        self.lst_elem = self.lst[:]
        self.lst_elem[x] = True

    def get_random_scv(self, obs):
        self.select_element(self.RANDOM_SCV_SELECTIONNE)
        unit_type = obs.observation["feature_screen"][UNIT_TYPE]
        scv_y, scv_x = (unit_type == units.Terran.SCV).nonzero()
        return [scv_x[0], scv_y[0]]


    def coordinate_patroll(self, obs):
        player_relative = obs.observation["feature_minimap"][features.SCREEN_FEATURES.player_relative.index]
        player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
        self.haut = player_y.mean() <= 31
        if self.haut:
            return [33, 33]
        return [33, 33]

    def coordinate_attack(self, obs):
        player_relative = obs.observation["feature_minimap"][features.SCREEN_FEATURES.player_relative.index]
        player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
        self.haut = player_y.mean() <= 31
        if self.haut:
            return [40, 45]
        return [20, 25]

    def select_CDC(self, obs):
        center = [unit for unit in obs.observation.feature_units if unit.unit_type == CENTRE_DE_COMMANDEMENT]
        if len(center) != 0:
            cdc = [center[0][coord_x], center[0][coord_y]]
            return cdc
        return []

    def select_caserne(self, obs):
        center = [unit for unit in obs.observation.feature_units if unit.unit_type == CASERNES]
        cdc = [center[0][coord_x], center[0][coord_y]]
        return cdc


    def update_units(self, obs):
        total_value_units = obs.observation["score_cumulative"][
                "total_value_units"]
        new_nb_scv = total_value_units / 50
        new_marines = obs.observation['player'][N_MARINES]
        diff = new_nb_scv - self.nombre_SCV
        diff_marines = new_marines - self.nombres_marines
        if diff_marines < 0:
            self.nombres_marines += diff_marines
        elif diff_marines > 0:
            self.nombre_marines_CEC -= diff_marines
        elif diff > 0:
            if self.nombre_scv_CEC > 0:
                self.nombre_scv_CEC -= diff
                self.premier_scv = True
        elif diff < 0:
            diff = new_nb_scv - self.nombre_SCV
            self.nombre_SCV += diff
        self.nombres_marines = new_marines

    def update_buildings(self, obs):
            total_value_structures = obs.observation["score_cumulative"][
                "total_value_structures"]
            if total_value_structures != self.structures:
                diff = (total_value_structures - self.structures)
                if diff == 100 and self.nombre_ravitaillement_CEC == 1:
                    self.nombre_ravitaillement += 1
                    self.nombre_ravitaillement_CEC -= 1
                if diff == 100 and self.nombre_bunker_CEC == 1:
                    self.nombre_bunker += 1
                    self.nombre_bunker_CEC -= 1
                if diff == 150 and self.nombre_caserne_CEC == 1:
                    self.nombre_caserne += 1
                    self.nombre_caserne_CEC -= 1
                elif diff == 125 and self.nombre_centreTechnique_CEC == 1:
                    self.nombre_centreTechnique += 1
                    self.nombre_centreTechnique_CEC-= 1

                self.structures = total_value_structures

    def select_marines(self, obs):
        marines = [unit for unit in obs.observation.feature_units if unit.unit_type == MARINES]
        return len(marines)

    def production_scv(self, obs):
        if (not self.is_elem_selected(self.CDC_SELECTIONNE) and actions.FUNCTIONS.select_point.id in obs.observation["available_actions"]):
            self.select_element(self.CDC_SELECTIONNE)
            cdc = self.select_CDC(obs)
            return actions.FUNCTIONS.select_point("select", cdc)
        if (self.is_elem_selected(self.CDC_SELECTIONNE)
                and actions.FUNCTIONS.Train_SCV_quick.id in obs.observation["available_actions"]):
            self.nombre_scv_CEC += 1
            return actions.FUNCTIONS.Train_SCV_quick("now")
        return actions.FUNCTIONS.no_op()

    def build_ravitaillement(self, obs, coord_x, coord_y):
        if not self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE):
            if not self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE):
                target = self.get_random_scv(obs)
                return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SCREEN, target])
        if (self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE)
                and actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.
                observation["available_actions"]):
            self.nombre_ravitaillement_CEC += 1
            target = [coord_x, coord_y]
            return actions.FunctionCall(actions.FUNCTIONS.Build_SupplyDepot_screen.id,
                                        [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()

    def build_raffinerie(self, obs, coord_x, coord_y):
        if not self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE):
            if not self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE):
                target = self.get_random_scv(obs)
                return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SCREEN, target])
        if (self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE)
                and actions.FUNCTIONS.Build_Refinery_screen.id in obs.
                observation["available_actions"]):
            self.nombre_raffinerie_CEC += 1
            target = [coord_x, coord_y]
            return actions.FunctionCall(actions.FUNCTIONS.Build_Refinery_screen.id,
                                        [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()

    def build_bunker(self, obs, coord_x, coord_y):
        if not self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE):
            if not self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE):
                target = self.get_random_scv(obs)
                return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SCREEN, target])
        if (self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE) and
                actions.FUNCTIONS.Build_Bunker_screen.id in obs.observation["available_actions"]):
            self.nombre_bunker_CEC += 1
            target = [coord_x, coord_y]
            return actions.FunctionCall(actions.FUNCTIONS.Build_Bunker_screen.id,
                                        [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()

    def build_caserne(self, obs, coord_x, coord_y):
        if not self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE):
            if not self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE):
                target = self.get_random_scv(obs)
                return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SCREEN, target])
        if (self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE)
                or self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE)):
            if actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation["available_actions"]:
                target = [coord_x, coord_y]
                self.nombre_caserne_CEC += 1
                return actions.FunctionCall(actions.FUNCTIONS.Build_Barracks_screen.id,
                                            [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()

    def build_centreTechnique(self, obs, coord_x, coord_y):
        if not self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE):
            if not self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE):
                target = self.get_random_scv(obs)
                return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SCREEN, target])
        if (self.is_elem_selected(self.SCV_INACTIF_SELECTIONNE)
                or self.is_elem_selected(self.RANDOM_SCV_SELECTIONNE)):
            if actions.FUNCTIONS.Build_EngineeringBay_screen.id in obs.observation["available_actions"]:
                target = [coord_x, coord_y]
                self.nombre_centreTechnique_CEC += 1
                return actions.FunctionCall(actions.FUNCTIONS.Build_EngineeringBay_screen.id,
                                            [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()

    def production_marine(self, obs):
        if not self.is_elem_selected(self.CASERNES_SELECTIONNE):
            if actions.FUNCTIONS.select_point.id in obs.observation["available_actions"]:
                self.select_element(self.CASERNES_SELECTIONNE)
                target = self.select_caserne(obs)
                return actions.FUNCTIONS.select_point("select", target)
        if actions.FUNCTIONS.Train_Marine_quick.id in obs.observation["available_actions"]:
            self.nombre_marines_CEC += 1
            return actions.FUNCTIONS.Train_Marine_quick("now")
        return actions.FUNCTIONS.no_op()

    def attack(self, obs):
        if self.mode_attaque == 0:
            if actions.FUNCTIONS.select_army.id in obs.observation["available_actions"]:
                self.mode_attaque = 1
                return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        if self.mode_attaque == 1:
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation["available_actions"]:
                player_relative = obs.observation["feature_screen"][
                    features.SCREEN_FEATURES.player_relative.index]
                marines_y, marines_x = (
                    player_relative == PLAYER_SELF).nonzero()
                if not marines_y.any():
                    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                self.initial_marines_x = numpy.sum(marines_x) / marines_x.size
                self.initial_marines_y = numpy.sum(marines_y) / marines_y.size
                dest = self.coordinate_attack(obs)
                return actions.FUNCTIONS.Attack_minimap("now", dest)
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def patroll(self, obs):
        if self.mode_patrouille == 0:
            if actions.FUNCTIONS.select_army.id in obs.observation["available_actions"]:
                self.mode_patrouille = 1
                return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        if self.mode_patrouille == 1:
            self.mode_patrouille = 2
            if actions.FUNCTIONS.Patrol_minimap.id in obs.observation["available_actions"]:
                player_relative = obs.observation["feature_screen"][
                    features.SCREEN_FEATURES.player_relative.index]
                marines_y, marines_x = (
                    player_relative == PLAYER_SELF).nonzero()
                if not marines_y.any():
                    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                self.initial_marines_x = numpy.sum(marines_x) / marines_x.size
                self.initial_marines_y = numpy.sum(marines_y) / marines_y.size
                dest = self.coordinate_patroll(obs)
                return actions.FUNCTIONS.Patrol_minimap("now", dest)
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])



    def step(self, obs):
        super(ProjetIA, self).step(obs)

        self.update_units(obs)
        self.update_buildings(obs)

        if not self.premier_scv and self.nombre_scv_CEC < 1:
            return self.production_scv(obs)

        if (self.premier_scv and self.nombre_ravitaillement == 0
                and self.nombre_ravitaillement_CEC < 1):
            player_relative = obs.observation["feature_minimap"][
                features.SCREEN_FEATURES.player_relative.index]
            player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
            cdc = self.select_CDC(obs)
            self.haut = player_y.mean() <= 31
            if self.haut:
                return self.build_ravitaillement(obs, cdc[0],
                                               cdc[1] + 20)
            return self.build_ravitaillement(obs, cdc[0],
                                           cdc[1] - 20)
        """
        if (self.premier_scv and self.nombre_raffinerie == 0 and self.nombre_raffinerie_CEC < 1 ):
            player_relative = obs.observation["feature_minimap"][
                features.SCREEN_FEATURES.player_relative.index]
            player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
            #cdc = self.target_gaz(obs)
            self.haut = player_y.mean() <= 31
            unit_type = obs.observation['feature_screen'][UNIT_TYPE]
            vespene_y, vespene_x =  (unit_type == 342).nonzero()
            i = random.randint(0, len(vespene_y)-1)

            xtest = vespene_x[i]
            ytest = vespene_y[i]

            print("JE SUIS RENTRE DANS LA FONCTION")


            if self.haut:
                return self.build_raffinerie(obs, xtest,
                                               ytest)
            return self.build_raffinerie(obs, xtest,
                                           ytest)

        """

        if (self.nombre_ravitaillement == 1 and self.nombre_centreTechnique == 0
                and self.nombre_centreTechnique_CEC < 1):
            player_relative = obs.observation["feature_minimap"][
                features.SCREEN_FEATURES.player_relative.index]
            player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
            cdc = self.select_CDC(obs)
            self.haut = player_y.mean() <= 31
            if self.haut:
                return self.build_centreTechnique(obs, cdc[0] + 35,
                                           cdc[1])
            return self.build_centreTechnique(obs, cdc[0] - 35, cdc[1])

        if (self.nombre_ravitaillement == 1 and self.nombre_centreTechnique == 1
                and self.nombre_ravitaillement_CEC < 1):
            player_relative = obs.observation["feature_minimap"][
                features.SCREEN_FEATURES.player_relative.index]

            player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
            cdc = self.select_CDC(obs)
            self.haut = player_y.mean() <= 31
            if self.haut:
                return self.build_ravitaillement(obs, cdc[0] + 10,
                                               cdc[1] + 20)
            return self.build_ravitaillement(obs, cdc[0] - 10,
                                           cdc[1] - 20)


        if (self.nombre_ravitaillement == 2 and self.nombre_centreTechnique == 1
                and self.nombre_ravitaillement_CEC < 1 and self.nombre_caserne == 0):
            player_relative = obs.observation["feature_minimap"][
                features.SCREEN_FEATURES.player_relative.index]
            player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
            cdc = self.select_CDC(obs)
            self.nombre_caserne += 1
            self.haut = player_y.mean() <= 31
            if self.haut:
                return self.build_caserne(obs, cdc[0] + 25,
                                               cdc[1] + 20)
            return self.build_caserne(obs, cdc[0] - 25,
                                           cdc[1] - 20)


        if (self.nombre_bunker == 0 and self.nombre_ravitaillement_CEC < 1
                and self.nombre_bunker_CEC < 1 and self.nombre_caserne == 1):
            player_relative = obs.observation["feature_minimap"][
                features.SCREEN_FEATURES.player_relative.index]
            player_y, player_x = (player_relative == PLAYER_SELF).nonzero()
            cdc = self.select_CDC(obs)
            self.haut = player_y.mean() <= 31
            if self.haut:
                return self.build_bunker(obs, cdc[0] + 30,
                                         cdc[1] + 10)
            return self.build_bunker(obs, cdc[0] - 30, cdc[1] - 10)


        if self.select_marines(obs) == 15 and self.phase_offensive == False:
            self.select_element(self._NONE)
            self.phase_offensive == True
            print("MODE PATROUILLAGE")
            return self.patroll(obs)


        """
        if self.select_marines(obs) > 15 :
            print("MODE ATTAQUE")
            self.select_element(self._NONE)
            return self.attack(obs)
        """


        if self.nombre_caserne > 0:
            print("Entrainement des marines")
            return self.production_marine(obs)
        return actions.FUNCTIONS.no_op()


def main(unused_argv):
    agent = ProjetIA()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[
                        sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.random,
                                    sc2_env.Difficulty.very_easy)
                    ],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(
                            screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=10,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
