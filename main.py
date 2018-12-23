import math

def run():
    return



def cost(x, x_target, u, t, T, cost_fn):
    # where T is max t
    if t == T: cost = terminal_cost(x, x_target, cost_fn)
    elif t<T: cost = interm_cost(u, cost_fn)
    else: assert(False)

    return cost


def terminal_cost(x, x_target, cost_fn):
    if cost_fn == 'first':
        cost = math.pow(x-x_target,2)

    else: assert(False)

    return cost

def interm_cost(u, cost_fn):
    if cost_fn == 'first':
        r=.00001
        cost = r*math.pow(u,2)
    else: assert(False)

    return cost



if __name__ == "__main__":
    run()