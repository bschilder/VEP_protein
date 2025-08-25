TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# this step will take ~3mins
import os 
import matplotlib.pyplot as plt 
from matplotlib.colors import to_hex
import pandas as pd
import numpy as np
import torch
import time
import string
import tqdm.notebook

import bokeh.plotting
bokeh.io.output_notebook()
from bokeh.models import BasicTicker, PrintfTickFormatter
from bokeh.palettes import viridis, RdBu
from bokeh.transform import linear_cmap
from bokeh.plotting import figure, show 


cmap = plt.colormaps["bwr_r"]
bwr_r = [to_hex(cmap(i)) for i in np.linspace(0, 1, 256)]
cmap = plt.colormaps["gray_r"]
gray = [to_hex(cmap(i)) for i in np.linspace(0, 1, 256)]

ALPHABET = "AFILVMWYDEKRHNQSTGPC"

def pssm_to_dataframe(pssm, esm_alphabet):
  sequence_length = pssm.shape[0]
  idx = [str(i) for i in np.arange(1, sequence_length + 1)]
  df = pd.DataFrame(pssm, index=idx, columns=list(esm_alphabet))
  df = df.stack().reset_index()
  df.columns = ['Position', 'Amino Acid', 'Probability']
  return df

def contact_to_dataframe(con):
  sequence_length = con.shape[0]
  idx = [str(i) for i in np.arange(1, sequence_length + 1)]
  df = pd.DataFrame(con, index=idx, columns=idx)
  df = df.stack().reset_index()
  df.columns = ['i', 'j', 'value']
  return df

def pair_to_dataframe(pair,esm_alphabet):
  sequence_length = pair.shape[0]
  df = pd.DataFrame(pair, index=list(esm_alphabet), columns=list(esm_alphabet))
  df = df.stack().reset_index()
  df.columns = ['aa_i', 'aa_j', 'value']
  return df


def load_model(model_name="esm2_t36_3B_UR50D", 
               DEVICE=DEVICE):
  
  if not os.path.isfile(f"/root/.cache/torch/hub/checkpoints/{model_name}.pt"):
    os.system(f"aria2c -q -x 16 -d /root/.cache/torch/hub/checkpoints/ https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt")
    os.system(f"aria2c -q -x 16 -d /root/.cache/torch/hub/checkpoints/ https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt")
  
  model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
  model = model.to(DEVICE)
  model = model.eval()
  return model, alphabet

def handle_parallel(seq, PARALLEL = 20):
  if PARALLEL is None:
    PARALLEL = 20
  if len(seq) > 1500:
    PARALLEL = 10
  elif len(seq) > 2400:
    PARALLEL = 1
  return PARALLEL

def get_logits(seq, 
               alphabet,
               model,
               PARALLEL=None, 
               return_jac=False,
               DEVICE=DEVICE,
               verbose=True):
  
  start_time = time.time()
  PARALLEL = handle_parallel(seq, PARALLEL=PARALLEL)

  if verbose:
    if return_jac:
      print(f"Computing logits and jacobians")
    else:
      print(f"Computing logits only")

    
  x,ln = alphabet.get_batch_converter()([(None,seq)])[-1],len(seq)
  with torch.no_grad():
    f = lambda x: model(x)["logits"][:,1:(ln+1),4:24].detach().cpu().numpy()
    logits = np.zeros((ln,20), dtype=np.float32)
    if return_jac:
      jac = np.zeros((ln,1,ln,20), dtype=np.float32)
      fx = f(x.to(DEVICE))[0]
    with tqdm.notebook.tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
      for n in range(0,ln,PARALLEL):
        m = min(n+PARALLEL,ln)
        x_h = torch.tile(torch.clone(x),[m-n,1])
        for i in range(m-n):
          x_h[i,n+i+1] = alphabet.mask_idx
        fx_h = f(x_h.to(DEVICE))
        for i in range(m-n):
          logits[n+i] = fx_h[i,n+i]
          if return_jac:
            jac[n+i] = fx_h[i,None] - fx
        pbar.update(m-n)
    
    if verbose:
      elapsed_time = time.time() - start_time
      print(f"Logits computed in {elapsed_time:.2f} seconds")
      
    if return_jac:
      return logits, jac
    else:
      return logits

def get_categorical_jacobian(seq, 
                             alphabet,
                             model,
                             layer=None, 
                             fast=False, 
                             DEVICE=DEVICE,
                             verbose=True):
  
  start_time = time.time()
  
  if verbose:
    print(f"Computing categorical jacobians")
  
  # ∂in/∂out
  x, ln = alphabet.get_batch_converter()([("seq", seq)])[-1], len(seq)
  with torch.no_grad():
    if layer is None:
      f = lambda x: model(x)["logits"][..., 1:(ln+1), 4:24].detach().cpu().numpy()
    else:
      f = lambda x: model(x, repr_layers=[layer])["representations"][layer][..., 1:(ln+1), :].detach().cpu().numpy()

    fx = f(x.to(DEVICE))[0]
    fx_h = np.zeros([ln, 1 if fast else 20, ln, fx.shape[-1]], dtype=np.float32)
    x = x.to(DEVICE) if fast else torch.tile(x, [20, 1]).to(DEVICE)
    with tqdm.notebook.tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
      for n in range(ln):  # for each position
        x_h = torch.clone(x)

        # mutate to all 20 aa
        x_h[:, n+1] = alphabet.mask_idx if fast else torch.arange(4, 24)
        fx_h[n] = f(x_h)
        pbar.update(1)

  # note: direction here differs from manuscript
  # positive = good
  # negative = bad
  
  if verbose:
    elapsed_time = time.time() - start_time
    print(f"Categorical jacobians computed in {elapsed_time:.2f} seconds")
    
  return fx_h - fx


def jac_to_con(jac, 
               ALPHABET_map,
               center=True, 
               diag="remove", 
               apc=True,
               symm=True):

  X = jac.copy()
  Lx,Ax,Ly,Ay = X.shape
  if Ax == 20:
    X = X[:,ALPHABET_map,:,:]

  if Ay == 20:
    X = X[:,:,:,ALPHABET_map]
    if symm and Ax == 20:
      X = (X + X.transpose(2,3,0,1))/2

  if center:
    for i in range(4):
      if X.shape[i] > 1:
        X -= X.mean(i,keepdims=True)

  contacts = np.sqrt(np.square(X).sum((1,3)))

  if symm and (Ax != 20 or Ay != 20):
    contacts = (contacts + contacts.T)/2

  if diag == "remove":
    np.fill_diagonal(contacts,0)

  if diag == "normalize":
    contacts_diag = np.diag(contacts)
    contacts = contacts / np.sqrt(contacts_diag[:,None] * contacts_diag[None,:])

  if apc:
    ap = contacts.sum(0,keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
    contacts = contacts - ap

  if diag == "remove":
    np.fill_diagonal(contacts,0)

  return {"jac":X, "contacts":contacts}

def get_alphabet_map(alphabet, ALPHABET=ALPHABET):
  esm_alphabet = list("".join(alphabet.all_toks[4:24]))
  ALPHABET_map = [esm_alphabet.index(a) for a in ALPHABET]
  return ALPHABET_map

def process_sequence(sequence):
  sequence = sequence.upper()
  sequence = ''.join([i for i in sequence if i.isalpha()])
  return sequence


def plot_logits(logits, 
                num_colors = 256,
                save_path=f"output/conservation_logits.txt",
                width=900, 
                height=400,
                TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
                 ): 

    from scipy.special import softmax
   
    
    # Save logits
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path,logits)
    
    # Convert logits to pssm
    pssm = softmax(logits,-1)
    df = pssm_to_dataframe(pssm, ALPHABET)

    # plot pssm 
    palette = viridis(num_colors)
   
    p = figure(title="CONSERVATION",
            x_range=[str(x) for x in range(1, logits.shape[0]+1)],
            y_range=list(ALPHABET)[::-1],
            width=width, 
            height=height,
            tools=TOOLS, 
            toolbar_location='below',
            tooltips=[('Position', '@Position'), ('Amino Acid', '@{Amino Acid}'), ('Probability', '@Probability')])

    r = p.rect(x="Position", 
               y="Amino Acid", 
               width=1, height=1, source=df,
            fill_color=linear_cmap('Probability', palette, low=0, high=1),
            line_color=None)
    p.xaxis.visible = False  # Hide the x-axis
    show(p)


def plot_jac(jac,
             ALPHABET_map,
             contacts_save_path=f"output/coevolution.txt",
             jac_save_path=f"output/jac.npy",
             fast = False, # @param {type:"boolean"}
            layer = None, # @param ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","None"] {type:"raw"}
            center = True, # @param {type:"boolean"}
            symm = True, # @param {type:"boolean"}
            diag = "remove", # @param ["remove", "normalize", "none"]
            apc = True, # @param {type:"boolean"}
            TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom",
            width=800, 
            height=800,
            ):
 
    con = jac_to_con(jac, 
                     ALPHABET_map=ALPHABET_map, 
                     center=center, 
                     diag=diag, 
                     apc=apc,
                     symm=symm)

    os.makedirs(os.path.dirname(contacts_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(jac_save_path), exist_ok=True)
    
    np.savetxt(contacts_save_path, con["contacts"])
    if layer is not None:
        i,j = np.triu_indices(jac.shape[0],1)
        np.save(jac_save_path,con["jac"][i,:,j,:].astype(np.float16))

    df = contact_to_dataframe(con["contacts"]) 
    
    p = figure(title="COEVOLUTION",
            x_range=[str(x) for x in range(1,jac.shape[0]+1)],
            y_range=[str(x) for x in range(1,jac.shape[0]+1)][::-1],
            width=width, 
            height=height,
            tools=TOOLS, toolbar_location='below',
            tooltips=[('i', '@i'), ('j', '@j'), ('value', '@value')])

    r = p.rect(x="i", y="j", width=1, height=1, source=df,
            fill_color=linear_cmap('value', gray, low=df.value.min(), high=df.value.max()),
            line_color=None)
    p.xaxis.visible = False  # Hide the x-axis
    p.yaxis.visible = False  # Hide the x-axis
    show(p)

    return con


def parse_fasta(filename, a3m=True):
  '''function to parse fasta file'''
  
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if line[0] == ">":
      header.append(line[1:])
      sequence.append([])
    else:
      if a3m: line = line.translate(rm_lc)
      else: line = line.upper()
      sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  
  return header, sequence
  
def mk_msa(seqs, alphabet=None):
  '''one hot encode msa'''
  if alphabet is None:
    alphabet = "ARNDCQEGHILKMFPSTWYV-"
  states = len(alphabet)  
  a2n = {a:n for n,a in enumerate(alphabet)}
  msa_ori = np.array([[a2n.get(aa, states-1) for aa in seq] for seq in seqs])
  return np.eye(states)[msa_ori]

def _do_apc(x, rm_diag=True):
  '''given matrix do apc correction'''
  if rm_diag: np.fill_diagonal(x,0.0)
  a1 = x.sum(0,keepdims=True)
  a2 = x.sum(1,keepdims=True)
  y = x - (a1*a2)/x.sum()
  np.fill_diagonal(y,0.0)
  return y

def inv_cov(Y):
  '''given one-hot encoded MSA, return contacts'''
  N,L,A = Y.shape
  Y_flat = Y.reshape(N,-1)
  c = np.cov(Y_flat.T)
  shrink = 4.5/np.sqrt(N) * np.eye(c.shape[0])
  ic = np.linalg.inv(c + shrink)
  raw = np.sqrt(np.square(ic.reshape(L,A,L,A)[:,:20,:,:20]).sum((1,3)))
  return {"c":c, "ic":ic,
          "raw":raw, "apc":_do_apc(raw)}

def inv_cov_torch(Y):
  Y = torch.tensor(Y, dtype=torch.float32)
  N,L,A = Y.shape
  Y_flat = Y.reshape(N,-1)
  c = torch.cov(Y_flat.T)
  shrink = 4.5/torch.sqrt(torch.tensor(N, dtype=torch.float32)) * torch.eye(c.shape[0], dtype=torch.float32)
  ic = torch.linalg.inv(c + shrink)
  raw = torch.sqrt(torch.square(ic.reshape(L,A,L,A)[:,:20,:,:20]).sum((1,3)))
  raw = raw.numpy()
  return {"c":c.numpy(),"ic":ic.numpy(),
          "raw":raw,"apc":_do_apc(raw)}


def do_apc(x, rm=1):
  '''given matrix do apc correction'''
  # trying to remove different number of components
  # rm=0 remove none
  # rm=1 apc
  x = np.copy(x)
  if rm == 0:
    return x
  elif rm == 1:
    a1 = x.sum(0,keepdims=True)
    a2 = x.sum(1,keepdims=True)
    y = x - (a1*a2)/x.sum()
  else:
    # decompose matrix, rm largest(s) eigenvectors
    u,s,v = np.linalg.svd(x)
    y = s[rm:] * u[:,rm:] @ v[rm:,:]
  np.fill_diagonal(y,0)
  return y

def get_contacts(x, symm=True, center=True, rm=1):
  # convert jacobian (L,A,L,A) to contact map (L,L)
  j = x.copy()
  if center:
    for i in range(4): j -= j.mean(i,keepdims=True)
  j_fn = np.sqrt(np.square(j).sum((1,3)))
  np.fill_diagonal(j_fn,0)
  j_fn_corrected = do_apc(j_fn, rm=rm)
  if symm:
    j_fn_corrected = (j_fn_corrected + j_fn_corrected.T)/2
  return j_fn_corrected


