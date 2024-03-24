import ROOT
import math
import numpy as np

def borel_pmf(k, lambda_):
    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / math.factorial(k)

def generate_random_borel(lambda_, max_k=25):
    # Generate PMF values
    pmf_values = [borel_pmf(k+1, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)


def simulate(init_photon=2,lambda_=0.2, p_ap=0.1):
    init_pe = np.random.poisson(init_photon)
    ct_pe = 0
    for _ in range(init_pe):
        ct_pe += 1 + generate_random_borel(lambda_)
    ap_pe = np.sum(np.random.rand(ct_pe) < p_ap)
    return init_photon, init_pe, ct_pe, ap_pe

def main():
    N_test = 1000000
    diff_list = []
    mu = 2
    lambda_ = 0.4
    ap = 0.08
    f1 = ROOT.TFile(f"single_test2.root","recreate")
    h_init_ph = ROOT.TH1F("init_ph", "initial photons", 100,0,100)
    h_init_pe = ROOT.TH1F("init_pe", "initial photon-electrons", 100,0,100)
    h_ct_pe = ROOT.TH1F("ct_pe", "crosstalk photon-electrons", 100,0,100)
    h_ap_pe = ROOT.TH1F("ap_pe", "afterpulse photon-electrons",100,0,100)
    h_total_pe = ROOT.TH1F("total_pe", "total photon-electrons", 10000,0,10000)
    for _ in range(N_test):
        init_ph, init_pe, ct_pe, ap_pe = simulate(init_photon=mu,lambda_=lambda_, p_ap=ap)
        h_init_ph.Fill(init_ph)
        h_init_pe.Fill(init_pe)
        h_ct_pe.Fill(ct_pe)
        h_ap_pe.Fill(ap_pe)
        h_total_pe.Fill((ct_pe + ap_pe) * 50)
    expression_value = (h_total_pe.GetRMS()**2 / h_total_pe.GetMean()**2) * mu - 1 / (1 - lambda_)
    print(mu, lambda_, ap, expression_value)
    # Prepare the string to be written
    output_string = f"{mu} {lambda_} {ap} {expression_value}\n"
    #with open('ap_map.txt', 'a') as file:
    #    file.write(output_string)
    diff_list.append((h_total_pe.GetRMS()**2/h_total_pe.GetMean()**2) * mu - 1/(1-lambda_))
    #sub_directory = f1.mkdir(f"mu_{round(mu,3)}_lambda_{round(lambda_,3)}_ap_{round(ap,4)}")
    #sub_directory.cd()

    h_init_ph.Write()
    h_init_pe.Write()
    h_ct_pe.Write()
    h_ap_pe.Write()
    h_total_pe.Write()
    f1.Close()

if __name__ == "__main__":
    main()
    


