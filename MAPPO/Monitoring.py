import matplotlib.pyplot as plt


def get_monitoring_weights_template():
    return {
        "fc1.weight": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc1.bias": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc2.weight": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc2.bias": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc3.weight": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc3.bias": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc4.weight": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
        "fc4.bias": {
            "norm": [],
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        },
    }


def get_monitoring_gradient_norm_template():
    return {
        "fc1.weight": [],
        "fc1.bias": [],
        "fc2.weight": [],
        "fc2.bias": [],
        "fc3.weight": [],
        "fc3.bias": [],
        "fc4.weight": [],
        "fc4.bias": [],
    }


# DEFINE GLOBALS
weight_monitoring_critic = get_monitoring_weights_template()
gradient_monitoring_critic = get_monitoring_gradient_norm_template()
weight_monitoring_actor0 = get_monitoring_weights_template()
gradient_monitoring_actor0 = get_monitoring_gradient_norm_template()
weight_monitoring_actor1 = get_monitoring_weights_template()
gradient_monitoring_actor1 = get_monitoring_gradient_norm_template()
loss_monitoring_total = []
loss_monitoring_critic = []
loss_monitoring_actor0 = []
loss_monitoring_actor1 = []


def visualize_all_graphs():
    visualize_network(gradient_monitoring_critic, weight_monitoring_critic, "Critic")
    visualize_network(gradient_monitoring_actor0, weight_monitoring_actor0, "Actor0")
    visualize_network(gradient_monitoring_actor1, weight_monitoring_actor1, "Actor1")
    visualize_losses()


def visualize_losses():
    plt.plot(loss_monitoring_total, label="Total Loss")
    plt.plot(loss_monitoring_critic, label="Critic Loss")
    plt.plot(loss_monitoring_actor0, label="Actor0 Loss")
    plt.plot(loss_monitoring_actor1, label="Actor1 Loss")
    plt.xlabel("Total Updates Count")
    plt.ylabel("Loss")
    plt.title("Loss Monitoring for Critic and Actors")
    plt.legend()
    plt.savefig("figures/loss_monitoring.png")
    plt.show()


def visualize_network(data_grad, data_weight, name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Larger size for better visibility
    fig.suptitle(f"Monitoring for {name}")

    ax = axes[0, 0]
    for key, values in data_grad.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Total Updates Count")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"Gradient Norms")

    stats = ["norm", "mean", "std", "max", "min"]
    for ind, s in enumerate(stats):
        ax = axes.flatten()[ind + 1]
        for key, value in data_weight.items():
            ax.plot(value[s], label=key)
        ax.set_xlabel("Total Updates Count")
        ax.set_ylabel(s)
        ax.set_title(f"Weight Stat {s}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles), fontsize=12)
    plt.savefig(f"figures/full_network_monitoring_{name}.png")
    plt.show()
