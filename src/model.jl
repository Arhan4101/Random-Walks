using Agents

# Random Walks and Diffusion

# Lets define our Agent, in 2 dimensional continuous space

@agent Walker{} ContinuousAgent{2} begin

end

# Lets define different Distributions for our Walkers

using Distributions

X = Distributions.Uniform(0, 1)

Y = Normal()

Z = Distributions.Levy(0,1)

W = Distributions.Exponential(1)

# Initialize the Walkers

function initialize(; n_agents = 100, extent = (100, 100))
    space = ContinuousSpace(extent; spacing = 0.1, periodic = true)
    scheduler = Schedulers.randomly
    model = AgentBasedModel(Walker, space; scheduler)

    # Adding the Walkers

    for i = 1:n_agents
        pos = (50, 50)
        θ = rand(model.rng, -π:π)
        vel = (cos(θ), sin(θ))

        agent = Walker(i, pos, vel)
        add_agent!(agent, model)
    end

    return model
end

# Stepping Function

function agent_step!(agent, model, D)
    α = rand(model.rng, -π:π)
    step_size = rand(D)
    agent.vel = (step_size).*(cos(α), sin(α))
    move_agent!(agent, model, 1.0)
end

# Animate Random Walkers using Makie

# Collect Data