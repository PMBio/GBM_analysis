###############################################
# topic_regulation.py
###############################################
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import sklearn

logger = logging.getLogger(__name__)

def get_activity_score(tf_activity, topics, low_exp_TFs=[]):
    """
    Get activity score of TFs in each topic
    
    Outputs:
        activity: Dataframe of activity score of each TF across provided set of topics.
    
    Args:
        tf_activity: Dataframe of TFs x Topics containing activity of each TF in each topic. TF activity is defined as
        the sum of all regulatory links towards target genes in each topic's eGRN. 
        topics: list of all topic numbers to consider. 
        low_exp_TFs: List of TFs to exclude due to low or sparse expression across cells (otherwise result in unfair 
            scaled weights across topics).
            
    """
    tf_activity=tf_activity[["Topic_"+str(t) for t in topics]]
    activity_across_TFs = (tf_activity - tf_activity.min())/(tf_activity.max() - tf_activity.min())
    activity_across_Topics = tf_activity.transpose()
    activity_across_Topics = (activity_across_Topics - activity_across_Topics.min())/(activity_across_Topics.max() - activity_across_Topics.min())
    activity_across_Topics = activity_across_Topics.transpose()
    activity = (activity_across_TFs+activity_across_Topics)/2
    activity.columns = topics
    activity=activity.fillna(0)
    if len(low_exp_TFs)>0:
        activity=activity.drop(low_exp_TFs)
        activity=(activity-activity.min())/(activity.max()-activity.min())

    
    return activity

def scale_gex_acc_topics(gex, acc, topics,TFs=[]):
    """
    Get scaled expression and accessibility values of TFs in each topic

    Outputs:
        gex: Dataframe of scaled average expression of each TF across provided set of topics.
        acc: Dataframe of scaled average accessibility (gene score) of each TF across provided set of topics.
    Args:
        gex: Dataframe of TFs x Topics containing average expression of each TF in each topic.
        acc: Dataframe of TFs x Topics containing average accessibility (gene score) of each TF in each topic.
        topics: list of all topics to consider (list of int values)
        TFs: List of TFs to consider.
            
    """
    if len(TFs)>0:
        gex = gex.loc[TFs]
        acc = acc.loc[TFs]
        
    gex=gex[["Topic_"+str(t) for t in topics]].transpose()
    gex = (gex - gex.min())/(gex.max() - gex.min())
    gex = gex.transpose()
    gex.columns = topics
    gex=gex.fillna(0)
    
    acc = acc[["Topic_"+str(t) for t in topics]].transpose()
    acc = (acc - acc.min())/(acc.max() - acc.min())
    acc = acc.transpose()
    acc.columns = topics
    acc=acc.fillna(0)
    
    return gex, acc
        

def compute_topic_regulation_potential(topics, activity, gex, acc, regulons, genes, low_exp_TFs=[], gex_thresh=0.5, TR_thresh=0.05, regulation="activation", use_exp=False, compute_significance=False, n=1000, percentile=90):
    """
    Compute activation/repression potential between pairs of topics.

    Outputs:
        topic_reg_matrix: Dataframe (source topics x target topics) containing TAP or TRS values 
            for each topic pair, normalized by total number of expressed TFs in source topic. If
            compute_significance is True, non-significant TAP/TRS values will be set to 0.
        tf_tr_links: Dataframe containing all regulatory links from TFs expressed in each source
            topic to TRs of each target topic. For each TR target, includes expression and 
            accessibility in source topic and activity score in target topic.
        random_background: If compute_significance is True, dataframe containing significance 
            thresholds for topic regulation values of every topic pair, as well as distribution 
            of randomized values used to compute these thresholds for every topic pair.
    
    Args:
        topics: list of strings with all topics to consider. Must match topic names in activity, 
            gex and acc dataframes.
        activity: Dataframe of TFs x Topics contaning activation score (if assessing TR regulation)
            or repression score (if assessing top repressor regulation) for each TF in each topic.
        gex: Dataframe of TFs x Topics containing average expression of each TF on the cells where
            each topic is active. Should be scaled across topics (values between 0 and 1)
        acc: Dataframe of TFs x Topics containing average accessibility of each TF on the cells where
            each topic is active. Should be scaled across topics (values between 0 and 1).
        regulons: 3-D numpy matrix ( Topics x TFs x Genes) with the regulon of each TF in each Topic.
        low_exp_TFs: TFs to exclude from analysis due to scattered expression or low expression in 
            all dataset. 
        gex_thresh: Expression threshold determining set of TFs expressed in cells of each topic.
            Default is 0.5.
        TR_thresh: Activity score threshold determining topic regulators (TRs) for each topic.
            Default is 0.05.
        regulation: Determines regulation to compute between topics. Must be either "activation"
            or "repression".
        use_exp: Boolean variable determining if expression of TRs should be used instead of 
            accessibility to assess TR priming.
        compute_significance: Boolean variable determining if significance of TAP/TRS values should
            be computed.  
        n: If compute_significance is True, determines number of times to compute randomized background.
            Default is 1000.
        percentile: Percentile of randomized TAP/TRS value distribution for each topic pair, above which
            real TAP/TRS values for the same topic pair are considered significant.
 
    
    """
    if regulation not in ("activation", "repression"):
        raise ValueError("Invalid regulation mode. Only 'activation' or 'repression' are allowed.")
    TFs = list(gex.index)
    #Remove lowly expressed TFs
    if len(low_exp_TFs)>0:
        #activity = activity.drop(low_exp_TFs)
        gex = gex.drop(low_exp_TFs)
        acc = acc.drop(low_exp_TFs)
        #Scale activity scores between 0 and 1 within each topic
        activity=(activity-activity.min())/(activity.max()-activity.min())
    
    #If assessing repression, get signatures of closed chromatin
    if regulation == "repression": 
        acc = 1-acc
        
    source_topic=[]
    target_topic=[]
    source_tf =[]
    target_tf=[]
    weight_s=[]
    weight_t =[]
    weight_val_act_score =[]
    weight_val_acc =[]
    weight_val_exp=[]
    weight_val_tot =[]
    logger.info("Computing regulatory interactions between topics")
    for topic_s in tqdm(topics):
        tf_s_exp = list(gex.index[gex[topic_s]>gex_thresh]) #Expressed TFs in source topic
        top_tfs_s = activity[topic_s].sort_values(ascending = False)
        top_tfs_s_vals=list(top_tfs_s) #Activity score of TRs of source topic
        top_tfs_s=[tf for tf,val in zip(top_tfs_s.index,top_tfs_s_vals) if val>TR_thresh] #TRs of source topic
        
        #Get GRN of source topic
        regulon_s = pd.DataFrame(regulons[int(topic_s),:,:],columns=genes)
        regulon_s.index = TFs
        regulon_s = regulon_s.transpose()
        
        for topic_t in topics:
            target = []
            source=[]
            weight_acc = []
            weight_exp=[]
            weight_act_score = []
            if topic_s != topic_t:
                top_tfs_e = activity[topic_t].sort_values(ascending = False)
                top_tfs_e_vals=list(top_tfs_e) #Activity score of TRs of target topic
                top_tfs_e=[tf for tf,val in zip(top_tfs_e.index,top_tfs_e_vals) if val>TR_thresh]
                for tf in tf_s_exp:   
                    regulon_s_tf = regulon_s[tf].copy() #target genes of TF
                    
                    if regulation == "activation":
                        #if TF in source topic is a TR of target topic, but not a TR of source topic, add recursive link
                        if (tf in top_tfs_e) and (tf not in top_tfs_s): 
                            source.extend([tf])
                            target.extend(["self"])
                            weight_acc.extend([acc[topic_s].loc[tf]])
                            weight_exp.extend([gex[topic_s].loc[tf]])
                            weight_act_score.extend([activity[topic_t].loc[tf]])
                        #activated TRs from target topic
                        overlap = [i for i in list(regulon_s_tf.index[regulon_s_tf>0]) if i in top_tfs_e] 
                    elif regulation == "repression":
                        #repressed TRs from target topic
                        overlap = [i for i in list(regulon_s_tf.index[regulon_s_tf<0]) if i in top_tfs_e] 
                    
                    #if any TR from target topic is targeted    
                    if len(overlap)>0: 
                        target.extend(overlap)
                        source.extend(list(np.repeat(tf,len(overlap))))
                        for val in overlap:
                            weight_acc.extend([acc[topic_s].loc[val]])
                            weight_exp.extend([gex[topic_s].loc[val]])
                            weight_act_score.extend([activity[topic_t].loc[val]])                            
                if len(target)>0:
                    source_tf.extend(source)
                    target_tf.extend(target)
                    source_topic.extend(list(np.repeat(topic_s,len(source))))
                    target_topic.extend(list(np.repeat(topic_t,len(source))))
                    weight_val_acc.extend(weight_acc)
                    weight_val_exp.extend(weight_exp)
                    weight_val_act_score.extend(weight_act_score)
                    if use_exp == False:
                        #total regulatory effect of TFs in source topic towards TRs in target topic
                        weight_val_tot.extend([sum(x*y for x, y in list(zip(weight_acc,weight_act_score)))])
                    else:
                        #total regulatory effect of TFs in source topic towards TRs in target topic, using expression to assess TR priming
                        weight_val_tot.extend([sum(x*y for x, y in list(zip(weight_exp,weight_act_score)))])
                else:
                    weight_val_tot.extend([0])
                weight_s.append(topic_s)
                weight_t.append(topic_t)
            else: #positive control - source and target topic are the same
                top_tfs_e = top_tfs_s
                for tf in tf_s_exp:
                    regulon_s_tf = regulon_s[tf] #target genes of TF
                    
                    if regulation == "activation":
                        if (tf in top_tfs_e):
                            source.extend([tf])
                            target.extend(["self"])
                            weight_exp.extend([gex[topic_s].loc[tf]])
                            weight_acc.extend([acc[topic_s].loc[tf]])
                            weight_act_score.extend([activity[topic_t].loc[tf]])
                        #activated TRs from own topic
                        overlap = [i for i in list(regulon_s_tf.index[regulon_s_tf>0]) if i in top_tfs_e] 
                    elif regulation == "repression":
                        #repressed TRs from own topic
                        overlap = [i for i in list(regulon_s_tf.index[regulon_s_tf<0]) if i in top_tfs_e] 

                    if len(overlap)>0:
                        target.extend(overlap)
                        source.extend(list(np.repeat(tf,len(overlap))))
                        for val in overlap:
                            weight_acc.extend([acc[topic_s].loc[val]])
                            weight_exp.extend([gex[topic_s].loc[val]])
                            weight_act_score.extend([activity[topic_t].loc[val]]) #Downstream effect                            
                if len(target)>0:
                    source_tf.extend(source)
                    target_tf.extend(target)
                    source_topic.extend(list(np.repeat(topic_s,len(source))))
                    target_topic.extend(list(np.repeat(topic_t,len(source))))
                    weight_val_acc.extend(weight_acc)
                    weight_val_exp.extend(weight_exp)
                    weight_val_act_score.extend(weight_act_score)
                    if use_exp==False:
                        weight_val_tot.extend([sum(x*y for x, y in list(zip(weight_acc,weight_act_score)))])
                    else:
                        weight_val_tot.extend([sum(x*y for x, y in list(zip(weight_exp,weight_act_score)))])
                else:
                    weight_val_tot.extend([0])
                weight_s.append(topic_s)
                weight_t.append(topic_t)                
    

    #Contribution of each TF-TR pair to TAP/TRS values
    tf_tr_links = pd.DataFrame({'Source_topic':source_topic, 'Target_topic':target_topic, 'TF':source_tf,
                                'TR':target_tf, 'Weight_acc':weight_val_acc, 'Weight_exp':weight_val_exp, 
                                'Weight_act_score':weight_val_act_score})

    #Activation/repression potential between each pair of topics
    topic_reg_matrix = np.array(weight_val_tot).reshape(len(topics),len(topics))
    topic_reg_matrix = pd.DataFrame(topic_reg_matrix, columns = topics)
    topic_reg_matrix.index = topics

    #Compute background activation/repression potential between pairs of topics using randomized eGRNs. 
    if compute_significance:
        topic_reg_matrix , random_background = compute_randomized_topic_regulation(topics, activity, gex, acc, regulons, genes, gex_thresh=gex_thresh, TR_thresh=TR_thresh, regulation=regulation, use_exp=use_exp, n=n, percentile=percentile)
    else:
        random_background=[]
        
    #Number of expressed TFs per topic
    num_exp_TFs_per_topic=pd.DataFrame((gex>gex_thresh).sum(axis=0))
    num_exp_TFs_per_topic = num_exp_TFs_per_topic.transpose()
    num_exp_TFs_per_topic.columns=topics
    
    #Normalize TAP/TRS values by number of TFs expressed in each source topic
    for t in topics:
        topic_reg_matrix.loc[t]=topic_reg_matrix.loc[t]/num_exp_TFs_per_topic[t][0]
        
    return topic_reg_matrix, tf_tr_links, random_background

def compute_randomized_topic_regulation(topics, activity, gex, acc, regulons, genes, gex_thresh=0.5, TR_thresh=0.05, regulation="activation", use_exp=False, n=1000, percentile=90):
    """
    Compute topic regulation potential with randomized GRNs to assess significance of real values.

    Outputs:
        topic_reg_matrix: Dataframe (source topics x target topics) containing TAP or TRS values 
            for each topic pair, normalized by total number of expressed TFs in source topic. If
            compute_significance is True, non-significant TAP/TRS values will be set to 0.
        random_background: Dataframe containing significance thresholds for topic regulation values
            of every topic pair, as well as distribution of randomized values used to compute these
            thresholds for every topic pair.

    Args:
        topics: list of strings with all topics to consider. Must match topic names in activity, 
            gex and acc dataframes.
        activity: Dataframe of TFs x Topics contaning activation score (if assessing TR regulation)
            or repression score (if assessing top repressor regulation) for each TF in each topic.
        gex: Dataframe of TFs x Topics containing average expression of each TF on the cells where
            each topic is active. Should be scaled across topics (values between 0 and 1)
        acc: Dataframe of TFs x Topics containing average accessibility of each TF on the cells where
            each topic is active. Should be scaled across topics (values between 0 and 1).
        regulons: 3-D numpy matrix ( Topics x TFs x Genes) with the regulon of each TF in each Topic.
        gex_thresh: Expression threshold determining set of TFs expressed in cells of each topic.
            Default is 0.5.
        TR_thresh: Activity score threshold determining topic regulators (TRs) for each topic.
            Default is 0.05.
        regulation: Determines regulation to compute between topics. Must be either "activation"
            or "repression".
        use_exp: Boolean variable determining if expression of TRs should be used instead of 
            accessibility to assess TR priming.
        n: If compute_significance is True, determines number of times to compute randomized background.
            Default is 1000.
        percentile: Percentile of randomized TAP/TRS value distribution for each topic pair, above which
            real TAP/TRS values for the same topic pair are considered significant.
    """
    
    logger.info("Computing significance of regulatory interactions")
    randomized_weights = np.empty(shape=[n,0])
    for topic_s in tqdm(topics):
        #Expressed TFs in source topic
        tf_s_exp = list(gex.index[gex[topic_s]>gex_thresh]) 
        
        #Get GRN of source topic
        regulon_s = pd.DataFrame(regulons[int(topic_s),:,:]).transpose()
        regulon_s.columns = TFs
        regulon_s.index = genes
        
        #Save true index and column values
        prev_ind = regulon_s.index
        prev_cols = regulon_s.columns
        
        #empty array to save background weights
        rand_weights = np.empty(shape=[0,len(topics)-1]) 
        for _ in tqdm(range(n)): #Compute background n times
            #shuffle rows of GRN
            regulon_s_shuff = regulon_s.iloc[np.random.permutation(len(regulon_s.index))].reset_index(drop=True) 
            regulon_s_shuff.index = prev_ind
            
            #shuffle columns of GRN
            shuff_cols = list(regulon_s.columns)
            random.shuffle(shuff_cols)
            regulon_s_shuff = regulon_s_shuff[shuff_cols]
            regulon_s_shuff.columns = prev_cols

            weight_val_tot =[]
            for topic_t in topics:
                top_tfs_e = activity[topic_t].sort_values(ascending = False)
                top_tfs_e_vals=list(top_tfs_e) #Activity score of TRs of target topic
                top_tfs_e=[tf for tf,val in zip(top_tfs_e.index,top_tfs_e_vals) if val>TR_thresh]
                
                target = []
                source=[]
                weight_acc = []
                weight_exp=[]
                weight_act_score = []
                for tf in tf_s_exp:   
                    regulon_s_shuff_tf = regulon_s_shuff[tf] #target genes of TF from shuffled GRN
                    
                    if regulation == "activation":
                        #activated TRs from target topic
                        overlap = [i for i in list(regulon_s_shuff_tf.index[regulon_s_shuff_tf>0]) if i in top_tfs_e] 
                    elif regulation == "repression":
                        #repressed TRs from target topic
                        overlap = [i for i in list(regulon_s_shuff_tf.index[regulon_s_shuff_tf<0]) if i in top_tfs_e] 
                        
                    if len(overlap)>0: #if any TR from target topic is targeted
                        target.extend(overlap)
                        source.extend(list(np.repeat(tf,len(overlap))))
                        for val in overlap:
                            weight_acc.extend([acc[topic_s].loc[val]])
                            weight_exp.extend([gex[topic_s].loc[val]])
                            weight_act_score.extend([activity[topic_t].loc[val]])                            
                if len(target)>0:
                    if use_exp == False:
                        #regulatory effect of TFs in source topic to TRs in target topic
                        weight_val_tot.extend([sum(x*y for x, y in list(zip(weight_acc,weight_act_score)))])
                    else:
                        #regulatory effect of TFs in source topic to TRs in target topic, using expression to assess TR priming
                        weight_val_tot.extend([sum(x*y for x, y in list(zip(weight_exp,weight_act_score)))])
                else:
                    weight_val_tot.extend([0])
                    
            #in each interation, save background TAP/TRS values from fixed source topic toward all other topics
            rand_weights = np.append(rand_weights,[np.array(weight_val_tot)], axis=0)
            
        #Append n background TAP/TRS values from fixed source topic to all other topics
        randomized_weights = np.append(randomized_weights,rand_weights,axis=1) 
        
    randomized_weights = pd.DataFrame(randomized_weights).transpose()
    #source_topic=[]
    #target_topic =[]
    #for s in topics:
    #    for e in topics:
    #        if s != e:
    #            source_topic.extend([s])
    #            target_topic.extend([e])
                
    topic_pairs = pd.DataFrame({'Source_topic':source_topic,'Target_topic':target_topic})
    random_background = pd.concat([topic_pairs, randomized_weights], axis=1)

    #Compute minimum TAP/TRS value for each topic pair above which true values are signficant 
    rand_background_dist = np.array(randomized_weights)
    percentile_vals = []
    for i in range(len(random_background)):
        p = np.percentile(rand_background_dist[i,:],percentile)
        percentile_vals.extend([p])
    random_background["Significance_threshold"] = percentile_vals
    ord_columns =['Source_topic','Target_topic','Significance_threshold']+[str(i) for i in range(n)]
    random_background = random_background[ord_columns]

    #Reshape background threshold values to match topic regulation potential
    background_mask = percentile_vals.reshape(len(topics),len(topics))
    background_mask = pd.DataFrame(background_mask, columns = topics)
    background_mask.index = topics

    #Remove non-significant TAP/TRS values 
    topic_reg_matrix[topic_reg_matrix<background_mask]=0  
    
    return topic_reg_matrix, random_background

def get_net_regulatory_interactions(topic_act_matrix, topic_rep_matrix, topics, random_act_background=[], random_rep_background=[], get_significant_links=True, percentile = 90):
    """
    Determines net regulatory interactions between topics.

    Outputs:
        net_topic_reg_matrix: Dataframe (source topics x target topics) containing TAP or 
            TRS values for each topic pair.
        
    Args:
        topic_act_matrix: Dataframe (source topics x target topics) containing TAP values
            for each topic pair, before significance testing.
        topic_rep_matrix: Dataframe (source topics x target topics) containing TRS values
            for each topic pair, before significance testing.
        random_act_background: Dataframe containing TAP values for each topic pair computed from multiple 
            randomizations of eGRNs. Output from compute_background_topic_regulation_potential().
        random_rep_background: Dataframe containing TRS values for each topic pair computed from multiple 
            randomizations of eGRNs. Output from compute_background_topic_regulation_potential().
        topics: list of all topics to consider. Topic numbers (int) must be provided.
        get_significant_links: Boolean value determining if non-significant links should be removed.
        percentile: Percentile of randomized TAP/TRS value distribution, for each topic pair, above which
            each real TAP/TRS values for the same topic pair are considered significant.

    """
    net_topic_reg_matrix = topic_act_matrix[topics].loc[topics]-topic_rep_matrix[topics].loc[topics]
    
    if get_significant_links:
        ids_act = [str(i) for i in range(len(random_act_background.columns)-3)]
        ids_rep = [str(i) for i in range(len(random_rep_background.columns)-3)]
        if len(ids_act)!= len(ids_rep):
            logger.info("Number of randomized background iterations is different for activation and repression matrices. Taking minimum value of iterations between the two.")
            if len(ids_act)>len(ids_rep):
                ids=ids_rep
            else:
                ids=ids_act
    
        random_act_background = random_act_background[ids]
        random_rep_background = random_act_background[ids]
        rand_background_dist = random_act_background-random_rep_background
        rand_background_dist = np.array(rand_background_dist)
        percentile_upper = []
        percentile_lower = []
        for i in range(len(rand_background_dist)):
            p_upper = np.percentile(rand_background_dist[i],percentile)
            p_lower = np.percentile(rand_background_dist[i],100-percentile)
            percentile_upper.extend([p_upper])
            percentile_lower.extend([p_upper])
    
        
        #Significance threshold for TAP-TRS values between each pair of topics
        background_mask_up = percentile_upper.reshape(len(topics),len(topics))
        background_mask_up = pd.DataFrame(background_mask_up, columns = topics)
        background_mask_up.index = topics
    
        background_mask_low = percentile_lower.reshape(len(topics),len(topics))
        background_mask_low = pd.DataFrame(background_mask_low, columns = topics)
        background_mask_low.index = topics
    
        #Only keep values above upper significance threshold OR below lower significance threshold
        net_topic_reg_matrix_u = net_topic_reg_matrix.copy()    
        net_topic_reg_matrix_u[net_topic_reg_matrix_u<background_mask_up]=0
    
        net_topic_reg_matrix_l = net_topic_reg_matrix.copy()    
        net_topic_reg_matrix_l[net_topic_reg_matrix_l>background_mask_low]=0

        net_topic_reg_matrix = net_topic_reg_matrix_u+net_topic_reg_matrix_l
    
    return net_topic_reg_matrix
        
    
def scale_topic_regulation_target_topic(topic_reg_matrix, scaling = "posCtrl", specificCtrl=[], return_scaling_vals = False):
    """
    Compute activation/repression potential between pairs of topics.

    Outputs:
        topic_reg_matrix: Dataframe (source topics x target topics) containing TAP or TRS values 
            for each topic pair scaled within target topic.
        scaling_vals_col: List of values used to scale each column of topic_reg_matrix. Returned
            only if return_scaling_vals is True.
        
    Args:
        topic_reg_matrix: Dataframe (source topics x target topics) containing TAP or TRS values
            for each topic pair, normalized by total number of expressed TFs in source topic.
        scaling: String determining which value to use for scaling all links targeting each topic
            (columns). Can be either "posCtrl" or "max". 
        specificCtrl: User provided values to scale links targeting each topic (columns). Must be 
            list of tuples indicating which value to use to scale each column.
            e.g.,  [(topicA, valA), (topicB,valB),...].

    """
    if scaling not in ("posCtrl", "max"):
        raise ValueError("Invalid scaling mode. Only 'posCtrl' or 'max' are allowed.")
    
    scaling_vals = []
    if len(specificCtrl)>0:
        scaled_col = []
        for target_t,val in specificCtrl:
            scaled_col.extend([target_t])
            scaling_vals.extend([val])
            topic_reg_matrix[target_t]=topic_reg_matrix[target_t]/val
        if len(scaled_col)<len(topic_reg_matrix.columns):
            unscaled_cols = [c for c in topic_reg_matrix.columns if c not in scaled_col]
            for t in unscaled_cols:
                if scaling == "posCtrl":
                    scaling_vals.extend([topic_reg_matrix[t].loc[t]])
                    topic_reg_matrix[t]=topic_reg_matrix[t]/topic_reg_matrix[t].loc[t]
                else:
                    scaling_vals.extend([topic_reg_matrix[t].max()])
                    topic_reg_matrix[t]=topic_reg_matrix[t]/topic_reg_matrix[t].max()
        else:
            unscaled_cols=[]
        cols=scaled_col+unscaled_cols
        scaling_vals_col = [(x,y) for x,y in zip(cols,scaling_vals)]
    else:
        if scaling == "posCtrl":
            for t in topic_reg_matrix.columns:
                scaling_vals.extend([topic_reg_matrix[t].loc[t]])
                topic_reg_matrix[t]=topic_reg_matrix[t]/topic_reg_matrix[t].loc[t]
        else:
            for t in topic_reg_matrix.columns:
                scaling_vals.extend([topic_reg_matrix[t].max()])
                topic_reg_matrix[t]=topic_reg_matrix[t]/topic_reg_matrix[t].max()
        scaling_vals_col = [(x,y) for x,y in zip(topic_reg_matrix.columns,scaling_vals)]

    if return_scaling_vals:
        return topic_reg_matrix, scaling_vals_col
    else:
        return topic_reg_matrix

def get_tf_topic_regulation(tf_tr_links, topics, scale_by_topic=True):
    """
    Get regulatory effect of individual TFs towards other topics.

    Outputs:
        tf_topic_regulation_ord: Dataframe of TFs x Topics containing cummulative regulatory 
            effect of every TF in source topic towards TRs in each target topic.
    Args:
        tf_tr_links: Dataframe containing all TF-TR links contributing to TAP or TRS values 
            between every topic pair, outputed from compute_topic_regulation_potential(). 
            Columns should contain: Source_topic, Target_topic, TF (in source topic), TR 
            (in target topic), Weight_acc, Weight_exp, Weight_act_score.
        topics: list of all topics to consider (list of int values)
        scale_by_topic: Boolean variable determining if contribution of each TF to topic 
            activation potential should be scaled across all TFs activating each topic. 
            Recommended if number of TRs is different across topics.
            
    """
    
    tfs_list=[]
    topic_list=[]
    tf_topic_regulation=pd.DataFrame(columns=topics)
    for topic in topics:
        #Dataframe with all TF-TR links originating from Source topic
        topic_df=tf_tr_links[tf_tr_links["Source_topic"]==topic]
        
        #TFs targeting TRs in other topics
        tfs_topic=list(set(list(tf_tr_links["TF"][tf_tr_links["Source_topic"]==topic]))) 
        tfs_list.extend(tfs_topic)
        topic_list.extend(list(np.repeat(topic,len(tfs_topic))))

        #Get cummulative regulatory effect of every TF in source topic towards TRs in each target topic
        aux=pd.DataFrame(columns=topics)
        for tf in tfs_topic:
            aux2=[]
            for t in topics:
                red = topic_df[(topic_df["Target_topic"]==t)&(topic_df["TF"]==tf)]
                aux2.extend([sum([x*y for x,y in zip(red["Weight_acc"],red["Weight_act_score"])])])
            aux=pd.concat([aux,pd.DataFrame([aux2],columns=topics)])

        tf_topic_regulation = pd.concat([tf_topic_regulation,aux])
    tf_topic_regulation.index=tfs_list
    
    #Scale TF regulatory effects within each topic
    if scale_by_topic:
        tf_topic_regulation=tf_topic_regulation/tf_topic_regulation.max()

    #Add info on source topic of each TF and total regulatory effect towards alternate topics
    tf_topic_regulation["Source_topic"]=topic_list
    tf_topic_regulation["Total_effect"] = tf_topic_regulation[topics].sum(axis=1)
    
    #Order TFs in each source topic by total activation effect towards alternate topics
    tf_topic_regulation_ord=pd.DataFrame(columns=tf_topic_regulation.columns)
    for topic in topics:
        aux=tf_topic_regulation[tf_topic_regulation["Source_topic"]==topic].sort_values(by="Total_effect",ascending=False)
        tf_topic_regulation_ord=pd.concat([tf_topic_regulation_ord,aux])
    ord_columns = ["Source_topic"] + topics + ["Total_effect"]
    tf_topic_regulation_ord=tf_topic_regulation_ord[ord_columns]
    
    return tf_topic_regulation_ord


def compute_state_level_regulation(topic_regulation_potential, topic_activity_per_state, topics, 
                                   scale_by_state=True, act_thresh=2/3, scale_by_val=[], return_scaling_vals=False):
    """
    Get total regulatory effect of topics in each cell state towards other states. 

    Outputs:
        state_regulation_potential: Dataframe (source states x target states) containing state
            activation or repression potential for each pair of states.
        scaling_vals: 
    Args:
        topic_regulation_potential: Dataframe (source topics x target topics) containing TAP or
            TRS values for each topic pair. Outputed from compute_topic_regulation_potential(). 
        topic_activity_per_state: Dataframe (States x Topics) containing average activation of 
            each topic within cells of each state.
        topics: list of all topics to consider (list of int values)
        act_thresh: Threshold on topic activity per state to determine relevant topics per state. 
            Must be relative to max activation of each topic in any state (value between 0 and 1).
            Default is 2/3.
        scale_by_state: Boolean variable determining if state regulation values should be scaled
            by each target state. If True, enables comparing the extent of activation or repression
            exerted by each state over the same target state.
        scale_by_val: User provided values to scale all state regulatory interactions towards each 
            state (columns). Must be list of tuples indicating which value to use to scale each column.
            e.g., [(stateA, valA), (stateB,valB),...].
        return_scaling_vals: Boolean variable determining if values used for scaling regulatory 
            interactions towards each state (columns) should be returned.
            
    """
    topic_activity_per_state= topic_activity_per_state[["Topic_"+str(t) for t in topics]]
    topic_activity_per_state.columns=topics
    
    #For each topic, only consider activity in states if it exceeds thresh. Meant to reduce noise.
    thresh = pd.DataFrame(topic_activity_per_state.max()*act_thresh).transpose()
    for t in topics:
        topic_activity_per_state[t][topic_activity_per_state[t]<thresh[t][0]]=0
    
    #Determine proportional activation of relevant topics within each state   
    topics_per_state=topic_activity_per_state.transpose()
    topics_per_state=topics_per_state/topics_per_state.sum()

    #Create mask keeping values from topics whose activity is specific to a state
    state_specific = topic_activity_per_state/topic_activity_per_state.sum()
    columns_to_keep = (state_specific == 1).any()
    state_specific.loc[:, ~columns_to_keep] = 0
    
    #Remove values from topics that are not state specific 
    state_specific_topics=(topic_activity_per_state*state_specific).transpose()
    #Determine proportional activation of all state-specific topics within each state
    state_specific_topics=state_specific_topics/state_specific_topics.sum()
    
    data_flat=np.array(topic_regulation_potential).flatten()
    source=[]
    target=[]
    for t in topics:
        source.extend(list(np.repeat(t,len(topic_regulation_potential.columns))))
        target.extend(list(topic_regulation_potential.columns))
    data_df=pd.DataFrame({"Source":source, "Target": target, "Link":data_flat})
    topics_per_state_dict={
        "Neuronal-like":list(topics_per_state["Neuronal-like"][topics_per_state["Neuronal-like"]>0].index),
        "OPC-NPC-like":list(topics_per_state["OPC-NPC-like"][topics_per_state["OPC-NPC-like"]>0].index),
        "AC-like":list(topics_per_state["AC-like"][topics_per_state["AC-like"]>0].index),
        "Gliosis-Hypoxia":list(topics_per_state["Gliosis-Hypoxia"][topics_per_state["Gliosis-Hypoxia"]>0].index),
        "Proliferative":list(topics_per_state["Proliferative"][topics_per_state["Proliferative"]>0].index)}
    state_specific_topics_dict = {
        "Neuronal-like":list(state_specific_topics["Neuronal-like"][state_specific_topics["Neuronal-like"]>0].index),
        "OPC-NPC-like":list(state_specific_topics["OPC-NPC-like"][state_specific_topics["OPC-NPC-like"]>0].index),
        "AC-like":list(state_specific_topics["AC-like"][state_specific_topics["AC-like"]>0].index),
        "Gliosis-Hypoxia":list(state_specific_topics["Gliosis-Hypoxia"][state_specific_topics["Gliosis-Hypoxia"]>0].index),
        "Proliferative":list(state_specific_topics["Proliferative"][state_specific_topics["Proliferative"]>0].index)}#v3
    
    
    state_regulation_potential = pd.DataFrame(columns=topics_per_state_dict.keys())
    for state_s,topics_s in topics_per_state_dict.items():
        pairwise_state_reg=[]
        for state_e,topics_e in state_specific_topics_dict.items():
            data_aux = data_df[data_df["Source"].isin(topics_s)]
            data_aux = data_aux[data_aux["Target"].isin(topics_e)]
            weights=[]
            for idx in data_aux.index:
                weights.extend([data_aux.loc[idx]["Link"]*topics_per_state[state_s].loc[data_aux.loc[idx]["Source"]]*state_specific_topics[state_e].loc[data_aux.loc[idx]["Target"]]])
            
            pairwise_state_reg.extend([sum(weights)])
        pairwise_state_reg = pd.DataFrame([pairwise_state_reg],columns=topics_per_state_dict.keys())
        state_regulation_potential = pd.concat([state_regulation_potential, pairwise_state_reg],axis=0)
    state_regulation_potential.index = topics_per_state_dict.keys()

    if len(scale_by_val)>0:
        for x,y in scale_by_val:
            state_regulation_potential[x]=state_regulation_potential[x]/y
        scaling_vals = scale_by_val  
    elif scale_by_state:
        scaling_vals = [(x,y) for x,y in zip(state_regulation_potential.columns, state_regulation_potential.max())]
        state_regulation_potential=state_regulation_potential/state_regulation_potential.max()

    if return_scaling_vals:
        return state_regulation_potential, scaling_vals
    else:
        return state_regulation_potential