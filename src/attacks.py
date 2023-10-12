import torch
import torch.nn.functional as F

def fgsm(input, epsilon, data_grad):
    perturbed_out = input + epsilon * data_grad.sign()

    return perturbed_out

def fgsm_attack(model, input, target, epsilon, criterion, max_iter = None):

    input_copy = input.clone().detach().requires_grad_(True)
    output = model(input_copy)

    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()

    data_grad = input_copy.grad.data

    perturbed_out = fgsm(input_copy, epsilon, data_grad)
    
    output = model(perturbed_out)

    input_copy = perturbed_out.clone().detach().requires_grad_(True)

    return perturbed_out


def ifgsm_attack(model, input, target, epsilon, criterion, max_iter = 10):
    input_copy = input.clone().detach().requires_grad_(True)
    epsilon = epsilon / max_iter
    for _ in range(max_iter):

        output = model(input_copy)

        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = input_copy.grad.data

        perturbed_out = fgsm(input_copy, epsilon, data_grad)
        
        output = model(perturbed_out)

        input_copy = perturbed_out.clone().detach().requires_grad_(True)

    return perturbed_out

def discriminator_output(models, input, lamb):
    out = 0
    for i in range(len(models)):
        out += F.sigmoid(models[i](input)).mean(-1)
    out = out * lamb / len(models)

    return out.sum()


def ifgsm_with_discr(model, discr, input, target, epsilon, criterion, lamb,  max_iter = 10):
    input_copy = input.clone().detach().requires_grad_(True)
    # epsilon = epsilon / max_iter
    for _ in range(max_iter):

        output = model(input_copy)
        
        loss = criterion(output, target) - discriminator_output(discr, input_copy, lamb)
        model.zero_grad()
        loss.backward()
        
        data_grad = input_copy.grad.data

        perturbed_out = fgsm(input_copy, epsilon, data_grad)
        
        output = model(perturbed_out)

        input_copy = perturbed_out.clone().detach().requires_grad_(True)

    return perturbed_out
    

attacks = {
    'fgsm': fgsm_attack,
    'ifgsm': ifgsm_attack,
    'ifgsm_discr': ifgsm_with_discr
}

def get_attack(attack = 'ifgsm'):

    return attacks[attack]
    