import pgmpy.models
import pgmpy.inference
import networkx as nx
import pylab as plt
import pgmpy.factors.discrete

if __name__ == "__main__":

    # Create a bayesian network
    model = pgmpy.models.BayesianModel([
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Earthquake', 'News'),
        ('Alarm', 'Watson')
    ])

    # Define conditional probability distributions (CPD)

    # Probability of burglary (True, False)
    cpd_burglary = pgmpy.factors.discrete.TabularCPD('Burglary', 2, [[0.7], [0.3]])

    # Probability of earthquake (True, False)
    cpd_earthquake = pgmpy.factors.discrete.TabularCPD('Earthquake', 2, [[0.65], [0.35]])

    # Probability of alarm going of (True, False) given a burglary and/or earthquake
    cpd_alarm = pgmpy.factors.discrete.TabularCPD(
        'Alarm', 2, [[0.95, 0.9, 0.6, 0.01], [0.05, 0.1, 0.4, 0.99]],
        evidence=['Burglary', 'Earthquake'], evidence_card=[2, 2]
    )

    # Probability that John calls (True, False) given that the alarm has sounded
    cpd_news = pgmpy.factors.discrete.TabularCPD(
        'News', 2, [[0.99, 0.4], [0.01, 0.6]],
        evidence=['Earthquake'], evidence_card=[2]
    )

    # Probability that Mary calls (True, False) given that the alarm has sounded
    cpd_mary = pgmpy.factors.discrete.TabularCPD(
        'Watson', 2, [[0.2, 0.6], [0.8, 0.4]],
        evidence=['Alarm'], evidence_card=[2]
    )

    # Add CPDs to the network structure
    model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_news, cpd_mary)
    # Check if the model is valid, throw an exception otherwise
    model.check_model()

    # Print probability distributions
    print('Probability distribution, P(Burglary)')
    print(cpd_burglary)
    print()
    print('Probability distribution, P(Earthquake)')
    print(cpd_earthquake)
    print()
    print('Joint probability distribution, P(Alarm | Burglary, Earthquake)')
    print(cpd_alarm)
    print()
    print('Joint probability distribution, P(News | Earthquake)')
    print(cpd_news)
    print()
    print('Joint probability distribution, P(MaryCalls | Alarm)')
    print(cpd_mary)
    print()

    # Plot the model
    nx.draw(model, with_labels=True)
    plt.savefig('alarm.png')
    plt.close()

    # Perform variable elimination for inference
    # Variable elimination (VE) is a an exact inference algorithm in bayesian networks
    infer = pgmpy.inference.VariableElimination(model)
    # Calculate the probability of a burglary if News and Watson calls (0: True, 1: False)
    posterior_probability = infer.query(['Earthquake'], evidence={'News': 1, 'Watson': 1})
    # Print posterior probability
    print('Posterior probability of Burglary if News(True) and Watson(True)')
    print(posterior_probability)
    print()
    # Calculate the probability of alarm starting if there is a burglary and an earthquake (0: True, 1: False)
    posterior_probability = infer.query(['Alarm'], evidence={'Burglary': 0, 'Earthquake': 0})
    # Print posterior probability
    print('Posterior probability of Alarm sounding if Burglary(True) and Earthquake(True)')
    print(posterior_probability)
    print()
