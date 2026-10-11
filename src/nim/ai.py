from __future__ import annotations

from typing import Literal

from nim import Action, Board, Probability, QValue

type Algorithm = Literal["greedy", "epsilon-greedy"]


class QLearning:
    def __init__(
        self,
        epsilon: Probability = 0.1,
        alpha: Probability = 0.5,
        gamma: Probability = 1.0,
    ):
        r"""Initialise l'IA.

        Une IA nouvellement initialisée possède
        un dictionnaire d'apprentissage Q vide,
        un taux d'apprentissage (alpha)
        et un taux d'exploration (epsilon).

        Args:
            epsilon:
                Le taux d'exploration.

                La chance de jouer un coup random lors de l'apprentissage (pas utilisé lors des vraies parties qui ne sont pas en mode apprentissage).
                Dans Q learning, on ne veut pas toujours trouver le "meilleur" coup à nos yeux.
                On veut plutot trouver le meilleur coup absolu, et non à nos yeux.
                C'est pour cela qu'on utilise epsilon. Utilisé lors du processus d'apprentissage, il nous permet de découvrir des nouveaux coups potentiels
                La raison est simple, si on jouait toujours le "meilleur" coup à nos yeux, on pourrait
                manquer des coups bien meilleurs. C'est comme quand tu vas à un resto et que puisqu'il est bon, tu y va tous le temps, alors tu en manques
                des bien meilleurs.
                Donc par exemple, un epsilon de 0.1 veut dire 10% de chance d'explorer un coup random.

                J'ai choisi 0.1, car je veux que ses coups fassent du sens et pour qu'il sache jouer contre des adversaires plus robustes
                Si j'avais choisi 1, il jouerait contre un adversaire qui n'utilise aucune logique tout comme lui,
                donc les résultats seraient basés sur des parties de mauvais joueurs (coups totalement randoms)
                TODO: epsilon (5 points)
                    Identifier une valeur par défaut
                    et documenter le paramètre alpha de manière à expliquer
                    le sens du paramètre et le choix de sa valeur.
            alpha:
                Le facteur d'apprentissage.
                Alpha sert seulement à définir la grosseur des modifications qu'il fait.
                Un alpha haut veut dire qu'il fait confiance à l'information qu'il apprend
                alors que un alpha bas veut dire qu'il ne fait pas totalement confiance à l'information qu'il apprend
                Dans d'autre mots, plus que tu fait confiance, plus que le changement du Q value va être drastique, ce qui
                n'est pas nécessairement bon puisque ca peut être erroné


                J'ai choisi 0.5 car de mon point de vue, 0.5 représente le milieu, ou dans d'autre mots, le point de balance.
                C'est-à-dire qu'il ne va pas totalement overwrite les vieilles valeurs, et il ne va pas apprendre trop lentement non plus

                TODO: alpha (5 points)
                    Identifier une valeur par défaut
                    et documenter le paramètre alpha de manière à expliquer
                    le sens du paramètre et le choix de sa valeur.
            gamma:
                Le facteur d'actualisation.

                Dans le Q-learning, γ (:math:`\gamma`) multiplie l'estimation
                de la valeur future optimale.
                L'importance accordée à la récompense suivante est définie
                par le paramètre gamma.

                Gamma est un nombre réel compris entre 0 et 1
                (:math:`0 \leq \gamma \leq 1`).
                Si l'on fixe gamma à zéro, l'agent ignore totalement les
                récompenses futures ;
                de tels agents ne prennent en compte que les récompenses
                immédiates.
                En revanche, si l'on fixe gamma à 1,
                l'algorithme recherche des récompenses élevées à long terme.


                J'ai choisi 1 car je veux qu'il donne autant d'influence aux récompenses immédiates qu'aux récompenses futures
                Aussi, vu que dans nim on peut pas retourner en arrière et que si on gagne, on gagne peu importe le nombre de coup,
                Je n'apporte pas d'influence aux coups potentiellement plus rapides puisque les deux amènent au même résultat.
                TODO: gamma (2 points)
                    Identifier une valeur par défaut


        Le dictionnaire d'apprentissage Q associe les paires (état, action) à
        une valeur-Q.

        """
        self.q: dict[tuple[Board, Action], QValue] = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def update(
        self, old_state: Board, action: Action, new_state: Board, reward: QValue
    ) -> None:
        """Met à jour le modèle Q-learning.

        Args:
            old_state:
                L'état avant l'action.
            action:
                L'action entreprise.
            new_state:
                L'état résultant de l'action.
            reward:
                La récompense obtenue.

        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state: Board, action: Action) -> QValue:
        """Retourne la valeur-Q pour un état et une action donnés.

        Si aucune valeur-Q n'existe encore dans `self.q`, retourne 0.
        """
        # TODO: get_q_value (10 points)
        #   Tester et implémenter get_q_value

    def update_q_value(
        self,
        state: Board,
        action: Action,
        old_q: QValue,
        reward: QValue,
        future_rewards: QValue,
    ) -> None:
        r"""Met à jour le Q-value.

        Args:
            state:
                L'état actuellement mis à jour.
            action:
                L'action qui met à jour l'état.
            old_q:
                L'ancienne valeur-Q pour cette paire état-action.
            reward:
                La récompense actuelle.
            future_rewards:
                L'estimation des récompenses futures.

        Utilise la formule:

        .. math::

            Q(S, A) <- Q_{old} + \alpha \cdot (Q_{new} - Q_{old})

        telle que :math:`Q_{old}` est l'ancienne Q-Value, :math:`\alpha` est le
        taux d'apprentissage, et :math:`Q_{new}` est la somme de la récompense
        actuelle et l'estimation des récompenses futures pondérée par le
        facteur d'actualisation.

        """
        # TODO: update_q_value (10 points)
        #   Tester et implémenter update_q_value
        raise NotImplementedError

    def best_future_reward(self, state: Board) -> QValue:
        """Estime la meilleure récompense future étant donné un état `state`.

        Considère toutes les paires `(état, action)` possibles disponibles
        dans cet état et retourne le maximum de toutes leurs valeurs-Q.
        Utilise 0 comme valeur-Q si une paire n'a pas de valeur-Q dans
        `self.q`.
        S'il n'y a aucune action disponible dans l'état, retourne 0.

        """
        # TODO: best_future_reward (10 points)
        #   Tester et implémenter best_future_reward
        raise NotImplementedError

    def choose_action(self, state: Board, train: bool = True) -> Action:
        """Étant donné un état, retourne une action à entreprendre.

        Si `train` est `False`, retourne la meilleure action disponible dans cet
        état (celle ayant la plus haute valeur-Q, en utilisant 0 pour les paires
        sans valeur-Q).

        Si `train` est `True`, choisit une action disponible aléatoire avec la
        probabilité `self.epsilon`, sinon choisit la meilleure action
        disponible.

        Si plusieurs actions ont la même valeur-Q, n'importe laquelle d'entre
        elles est une valeur de retour acceptable.

        """
        # TODO: choose_action (10 points)
        #   Tester et implémenter choose_action
        raise NotImplementedError
