
  L'architettura separa nettamente la fase di calibrazione e generazione dei template di riferimento (Flow A) dalla fase di rilevamento, classificazione e conteggio sul target   
  set (Flow B), affidandosi esclusivamente a tecniche di Computer Vision classica.                                                                                                
  ──────                                                                                                                                                                          
  ### 1. Flow_A_Reference: Calibrazione Offline e Feature Templates                                                                                                               
                                                                                                                                                                                  
  Il suo scopo è estrarre le "impronte digitali" cromatiche e strutturali delle 8 monete note del reference_set (1c, 2c, 5c, 10c, 20c, 50c, 1€, 2€) ed è orchestrato da           
  flow_A_reference.ipynb:                                                                                                                                                         
                                                                                                                                                                                  
  1. Caricamento: module1_load_references.py carica le 8 immagini di riferimento con etichette note.                                                                              
  2. Normalizzazione ROI: module2_extract_roi.py estrae ROI centrate e scalate a risoluzione fissa (250×250 px con raggio standard R = 100 px).                                   
  3. Calibrazione Cromatiche: module3_extract_color.py definisce e salva le soglie decisionali in color_profiles.json:                                                            
      • Bimetalliche: gradiente Δb^* nello spazio Lab (anello esterno vs core centrale) per isolare 1€ e 2€.                                                                      
      • Monometalliche: White-Patch per la costanza cromatica e soglia su Tonalità (Hue) per separare monete in rame (copper: 1c, 2c, 5c) e ottone/nordic gold (gold: 10c, 20c,   
      50c).                                                                                                                                                                       
  4. Estrazione Bordi: module4_extract_edges.py estrae i bassorilievi e contorni tramite CLAHE, Bilateral Filter, Canny Edge Detection e mascheramento circolare (R ≤ 99 px).     
  5. Polar Unwarp (Template Generation): module5_polar_unwarp.py applica Gaussian blur e srotola la mappa dei bordi in coordinate polari (θ, r) tramite cv2.warpPolar producendo  
  matrici normalizzate a 360 bins angolari × 50 bins radiali (salvate come file .npy / .npz).                                                                                     
  ──────                                                                                                                                                                          
  ### 2. Flow_B_Target: Detection, Matching e Valutazione                                                                                                                         
                                                                                                                                                                                  
  Il suo scopo è elaborare le immagini "in the wild" del target_set utilizzando i parametri e i template generati dal Flow A. È orchestrato da flow_b_target.ipynb:               
                                                                                                                                                                                  
  1. Caricamento: module1_load_target.py carica le immagini target.                                                                                                               
  2. Denoising per Hough: module2_denoise_for_detection.py applica canale L (LAB), Median Blur, Fast Non-Local Means (NLM) per rimuovere la texture di fondo e stabilizzare i     
  gradienti.                                                                                                                                                                      
  3. Circle Detection: module3_coin_detection.py esegue la Hough Circle Transform con filtro di validazione sul supporto radiale dei gradienti (edge support) per eliminare falsi 
  positivi, memorizzando i cerchi trovati nel manifest centrale annotated_images.json.                                                                                            
  4. Denoising a Colori: module4_denoise_colored.py riduce il rumore cromatico con fastNlMeansDenoisingColored + filtro mediano 3×3.                                              
  5. Classificazione Cromatica (Pruning): module5_coin_color_classification.py assegna a ogni moneta una classe cromatica (kind):                                                 
      • 1 euro e 2 euro vengono identificate direttamente;                                                                                                                        
      • copper e gold restringono i candidati a 3 tagli ciascuna ({1c, 2c, 5c} o {10c, 20c, 50c}).                                                                                
  6. Polar Unwarp Target: module6_polar_unwarp.py normalizza ed estrae la matrice polare (360×50) per ogni moneta target, in maniera identica a Flow A.                           
  7. Classificazione Disgiunta (Singola vs Multipla):                                                                                                                             
      • Moneta Singola (module7_1_single_coin_classification.py): Non avendo riferimenti di scala relativa, usa Circular ZNCC (Zero-Normalized Cross-Correlation): confronta la   
      matrice polare della moneta target con i template dei candidati di Flow A lungo tutti i 360 shift angolari per garantire l'invarianza a rotazione.                          
      • Monete Multiple (module7_2_multi_coin_classification.py): Sfrutta l'invarianza di scala relativa e formula un problema CSP (Constraint Satisfaction Problem): poiché tutte
      le monete della scena condividono la stessa distanza focale/piano, il fattore S = rₚₓ/Rₙₒₘᵢₙₐₗₑ deve essere costante; si ottimizza la tupla globale minimizzando la varianza
      di scala e rispettando i vincoli di monotonicità e confidenza cromatica.                                                                                                    
  8. Visualizzazione e Somma Monetaria: module8_visualize_results.py traccia bounding box e etichette, calcola il parziale per immagine e l'ammontare complessivo                 
  (final_counting_summary.json).                                                                                                                                                  
  9. Valutazione Quantitativa: module9_evaluation.py calcola metriche rispetto al Ground Truth (annotations_with_position.csv) tramite bipartite matching (algoritmo ungherese),  
  fornendo Precision/Recall/F1, matrice di confusione 8×8 e MAE monetario.