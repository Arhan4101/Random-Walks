using Agents

# Random Walks and Diffusion

# Lets define our Agent, in 2 dimensional continuous space

@agent Walker{} ContinuousAgent{2} begin
    Group::Int8
end

# Lets define different Distributions for our Walkers

using Distributions

Unif = Distributions.Uniform(0, 1)

Gauss = Normal()

Levy = Distributions.Levy(0,1)

Exp = Distributions.Exponential(1)

Dists = [Unif, Gauss, Levy, Exp]

# Initialize the Walkers

function initialize(; n_agents = 1, extent = (100, 100))
    space = ContinuousSpace(extent; spacing = 0.1, periodic = true)
    scheduler = Schedulers.randomly
    model = AgentBasedModel(Walker, space; scheduler)

    # Adding the Walkers

    for i = 1:n_agents
        pos = (50,50)
        θ = rand(model.rng, -π:π)
        vel = (cos(θ), sin(θ))
        Group = rand(model.rng, 1:4)

        agent = Walker(i, pos, vel, Group)
        add_agent_pos!(agent, model)
    end

    return model
end

# Stepping Function

function agent_step!(agent, model)
    α = rand(model.rng, -π:π)
    step_size = rand(Dists[agent.Group])
    agent.vel = (step_size).*(cos(α), sin(α))
    move_agent!(agent, model, 1.0)
end

# Animate Random Walkers using Makie


# Collect Data

function Get_Data(n_steps)
    adata = [:pos, :vel, :Group]

    model = initialize()
    data, _ = run!(model, agent_step!, n_steps; adata)
end

