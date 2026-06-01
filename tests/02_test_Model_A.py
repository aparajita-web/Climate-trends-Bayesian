import numpy as np
sys.path.append("../src/climate_trends/")
from Model_A import log_likelihood,log_probability,climate_model


def test_climate_model():

    theta = [1.0, 2.0, 0.5, 1.0]
    t = np.array([0, 1, 2])

    result = climate_model(theta, t)

    expected = theta[0]+theta[1]*t

    np.testing.assert_allclose(result, expected)


### Test if we the log likelihood returns expected values when $phi$=0
def test_log_likelihood():

    t = np.array([0, 1, 2])

    theta = [1.0, 2.0, 0.0, 1.0]

    y = climate_model(theta, t)

    ll = log_likelihood(theta, t, y)

    expected = -0.5 * 2 * np.log(2*np.pi)

    np.testing.assert_allclose(ll, expected)


