import re

def getDamageScenario(path) :
    """
    Take path variable .txt and return int associate in the end of file name.
    Exemple:
    input: ./test/banana123.txt
    return: 123
    """

    regex = re.compile(r'\d+')
    if (path[-5] != 'U'):
        scenario_number = [int(x) for x in regex.findall(path)];
    else :
        scenario_number = [0];

    return scenario_number[0];
