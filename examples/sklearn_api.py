from sklearn.datasets import make_regression
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split
from cartesian import Symbolic

rng = check_random_state(1337)
x, y, coef = make_regression(
    n_features=2, n_informative=1, n_targets=1, random_state=rng, coef=True
)


def log(res):
    if res.nit % 2 == 0:
        print(f"Generation: {res.nit}\tCurrent best: {res.expr}")


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng)
est = Symbolic(
    n_const=2,
    random_state=rng,
    n_columns=5,
    n_rows=2,
    maxfev=100000,
    n_jobs=1,
    f_tol=1e-4,
    callback=log,
)
est.fit(x_train, y_train)
print("Final result:")
print(est.res)
print("Test score: ", est.score(x_test, y_test))
