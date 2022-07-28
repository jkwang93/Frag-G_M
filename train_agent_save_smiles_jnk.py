#!/usr/bin/env python
import warnings

import torch
import pickle
import numpy as np
import pandas as pd
import time
import os
from shutil import copyfile

from model import RNN
from data_structs_fragment import Vocabulary, Experience
# from scoring_functions import get_scoring_function
from properties import multi_scoring_functions_one_hot_dual
from properties import get_scoring_function, qed_func, sa_func
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, seq_to_smiles_frag
from vizard_logger import VizardLog

warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_agent(epoch, restore_prior_from='/apdcephfs/private_jikewang/W4_reduce_RL/data/6w_Prior_fragment.ckpt',
                restore_agent_from='/apdcephfs/private_jikewang/W4_reduce_RL/data/6w_Prior_fragment.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=128, n_steps=5001,
                num_processes=0, sigma=60,
                experience_replay=0):
    voc = Vocabulary(init_from_file="/apdcephfs/private_jikewang/W4_reduce_RL/data/fragments.csv")

    logger = VizardLog('/apdcephfs/private_jikewang/W4_reduce_RL/data/logs')

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from,map_location={'cuda:0':'cuda:0'}))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0001)

    # Scoring_function

    # scoring_list = ['jnk3', 'gsk3', 'qed', 'sa']

    #
    # print(scoring_function)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")

    # Scoring_function
    scoring_function1 = get_scoring_function('jnk3')
    scoring_function2 = get_scoring_function('gsk3')
    smiles_save = []
    expericence_step_index = []
    score_list = []
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size=batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood,_ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles_frag(seqs, voc)

        score1 = scoring_function1(smiles)
        score2 = scoring_function2(smiles)
        qed = qed_func()(smiles)
        sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
                      dtype=np.float32)  # to keep all reward components between [0,1]
        score = score1 + score2+qed + sa
        score_list.append(np.mean(score))
        # 判断是否为success分子，并储存
        success_score = multi_scoring_functions_one_hot_dual(smiles, ['jnk3','gsk3', 'qed', 'sa'])
        itemindex = list(np.where(success_score == 4))
        success_smiles = np.array(smiles)[itemindex]
        smiles_save.extend(success_smiles)
        expericence_step_index = expericence_step_index + len(success_smiles) * [step]

        # TODO
        if step >= 5000:
            print('num: ', len(set(smiles_save)))
            save_smiles_df = pd.concat([pd.DataFrame(smiles_save), pd.DataFrame(expericence_step_index)], axis=1)
            save_smiles_df.to_csv('/apdcephfs/private_jikewang/W4_reduce_RL/output/jnk_rl00001/' + str(epoch) + '_frag_jnk_5w.csv', index=False, header=False)
            pd.DataFrame(score_list).to_csv('/apdcephfs/private_jikewang/W4_reduce_RL/output/jnk_rl00001/' + str(epoch) + '_chembl_frag_jnk_mean_score.csv', index=False, header=False)
            break

        # score = multi_scoring_functions(smiles, scoring_list)
        # score = multi_scoring_functions(smiles, scoring_list)

        # funcs = [scoring_function(prop) for prop in args.prop.split(',')]
        # score = scoring_function(smiles)
        # print(prior_likelihood)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
            step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        try:
            for i in range(10):
                print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        except Exception as e:
            print (e)
        # # Need this for Vizard plotting
        # step_score[0].append(step + 1)
        # step_score[1].append(np.mean(score))
        #
        # # Log some weights
        # logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        # logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        # logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        # logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        # logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        # logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
        #     (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        # logger.log(np.array(step_score), "Scores")

        # If the entire training finishes, we create a new folder where we save this python file
        # as well as some sampled sequences and the contents of the experinence (which are the highest
        # scored sequences seen during training)
        # if step == 50:
        #     torch.save(Agent.rnn.state_dict(), 'no_one_hot_50_Agent.ckpt')
        # if step%10==0 and step!=0:
        #     torch.save(Agent.rnn.state_dict(), './yuzhi/only_gsk/'+str(step)+'.ckpt')




    # if not save_dir:
    #     save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    # os.makedirs(save_dir)
    # copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))
    #
    # experience.print_memory(os.path.join(save_dir, "memory"))
    # torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'MC_MIT_301_Agent.ckpt'))
    #
    # seqs, agent_likelihood, entropy = Agent.sample(256)
    # prior_likelihood = Prior.likelihood(Variable(seqs))
    # prior_likelihood = prior_likelihood.data.cpu().numpy()
    # smiles = seq_to_smiles(seqs, voc)
    # # score = multi_scoring_functions(smiles, scoring_list)
    # score1 = scoring_function1(smiles)
    # # score2 = scoring_function2(smiles)
    # qed = qed_func()(smiles)
    # sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
    #               dtype=np.float32)  # to keep all reward components between [0,1]
    # score = score1 + score2 + qed + sa
    # # score = multi_scoring_functions(smiles, scoring_list)
    #
    # with open(os.path.join(save_dir, "sampled"), 'w') as f:
    #     f.write("SMILES Score PriorLogP\n")
    #     for smiles, score, prior_likelihood in zip(smiles, score, prior_likelihood):
    #         f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score, prior_likelihood))


if __name__ == "__main__":
    for i in range(5):
        train_agent(epoch=i)
