(**********************************************************************)
(* Simulation de l'épidémie de covid19 en France *)
(**********************************************************************)
(* si arguments: 
1: le nom du répertoire ou mettre les fichiers (si rien: france_rea, si quelque chose,
 on ne met pas à jour la synthèse)
2: le jour où arrêter de prendre en compte les données (si rien: le max)
3: le nombre de jours de la simulation (si rien, elle s'arrête 3 jours après le présent)
4: le fichier des estimations (si rien, rien...)

ex:
pour calculer les limites: 
time ./propage france_rea_limite 60 500 france_rea_jour_60/_estimations.csv
*)

let pays = ref (try Sys.argv.(1) with _ -> "france_rea");;

(**********************************************************************)
(* utilitaires *)

let f_i = float_of_int;;
let i_f = int_of_float;;
open Printf;;
let pr = Printf.printf;;
Random.self_init();;
(* c erreur relative *)
let hasard_float m c =
  let err = c *. m in
  m -. err +. 2. *. (Random.float err)
;;
let hasard_float_intervalle a b =
  a +. Random.float (b -. a)
;;
(* de m à m + err *)
let hasard_int m err =
  max 0 (m - err + Random.int (2*err + 1))
;;
(* de a à b *)
let hasard_int_intervalle a b =
  a + Random.int (b - a + 1)
;;

let read_file filename = 
  let lines = ref [] in
  let chan = open_in filename in
  try
    while true do
      lines := input_line chan :: !lines
    done;
    !lines
  with End_of_file ->
    close_in chan;
    List.rev !lines
;;

let explode str =
  let rec exp a b =
    if a < 0 then b
    else exp (a - 1) (str.[a] :: b)
  in
  exp (String.length str - 1) []
;;

let rec implode l =
  match l with [] -> "" | c::q -> (String.make 1 c) ^(implode q)
;;

let split s delim_char =
  let rec split input curr_word past_words =
    match input with
      | [] -> curr_word :: past_words
      | c :: rest ->
        if c = delim_char
        then split rest [] (curr_word :: past_words)
        else split rest (c :: curr_word) past_words
  in
  List.rev (List.map (fun x -> implode (List.rev x)) (split (explode s) [] []))
;;

(***********************************************************************)
(* patients covid en réanimation/soins intensifs *)

(* https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/
   https://geodes.santepubliquefrance.fr/#c=indicator&f=0&i=covid_hospit.rea&s=2020-04-25&t=a01&view=map2 
   https://www.gouvernement.fr/info-coronavirus/carte-et-donnees
*)
let france_rea =
  [|15;23;39;45;55;
    (* 9 mars, rassemblements > 1000 interdits, geste barrières *)
    66;86;105;129;210;300;400;550;
    (* confinement: 17 mars *)
    699;
    (* 18 mars *)
    771;1002;1297;1453;1674;2080;2503;2935;3351;3758;4236;4592;5056;5496;
    (* 1 avril *)
    5940;6305;6556;6723;6859;6948;7004;7019;6937;6875;
    (* 11 avril *)
    6752;6714;6690;6599;6331;6139;5922;5733;5644;5584;
    (* 21 avril *)
    5334;5127;4967;4785;4641;4598;4526;4392;4128;3947;
    (* 29 et 30 avril: arrêt de Geodes,  sur le site du gouv 4207;4019;*)
    (* 1er mai *)
    3819;3770;3762;3639;3375;3095
  |]
;;

(* rea cumulés *)
let s = ref 0;;
for i = 0 to Array.length france_rea - 1 do
  s := !s + france_rea.(i);
  france_rea.(i) <- !s;
done;;

(* on extrapole le nombre de morts: environ 8% des rea, 
 on oublie les ehpad (conditions de transimission spéciales)
 pour avoir une valeur potable de la mortalité*)
let france_rea = Array.map
                   (fun x -> i_f (f_i x *. 0.073 (* *. 19323. /. 11842. *) ))
                   france_rea;;
(* décalage de 1 jour dans les bilans journaliers : on a les chiffres de la veille *)
let confinement_france_rea = 1 + 13;; (* mardi  17 mars, 699 en réa *)
let decal_deconfinement = ref 1000 ;; 
let debut_france_rea = "March 4 2020";;

(* la simulation s'arrête au présent *)
let arret_au_present = ref true;;
let lejour = try int_of_string Sys.argv.(2) with _ -> Array.length france_rea;;
let france_rea = Array.sub france_rea 0 lejour;;
(***********************************************************************)
(* paramètres propres au pays *)

let pays_morts = ref france_rea;;
let debut_confinement = ref confinement_france_rea;;
let debut_pays = ref debut_france_rea;;

(* population *)

let million = 1000000;;
let population = 60*million;;

let voisins = 25*1000*13 ;; (* 1000 * durée infectant *)
let largeur_pays = 700 ;; (* largeur en kilometres du carré représentant le pays, 
                  même surface que la France *)

let proba_voyageur = ref 0.10;;
let div_proba_voyageur = ref 100.;; (* facteur de réduction après le confinement *)
let div_dvoisins = ref 10;; (* facteur de réduction après le confinement *)

(* facteur de réduction pour les pixels de la carte de l'épidémie *)
let red_carte = 50 ;;

let jourfr = Array.length !pays_morts;;
let nombre_de_jours_a_simuler = (try arret_au_present := false; int_of_string Sys.argv.(3)
                                 with _ -> (arret_au_present := true; 500));;
let dir = let dir0 = (try Sys.argv.(1) with _ -> !pays) ^ "_jour_" ^ string_of_int jourfr in
          if !arret_au_present
          then dir0
          else dir0 (* ^ "_limite"*)
;;
let _ = Sys.command ("mkdir " ^ dir);;
let file_estimations = (try Sys.argv.(4) with _ -> "");;
                         
(***********************************************************************)
(* paramètres de la maladie
sources:
 https://fr.wikipedia.org/wiki/Maladie_à_coronavirus_2019 
 https://www.lemonde.fr/blog/realitesbiomedicales/2020/04/17/covid-19-interrogations-sur-lexcretion-du-virus-et-la-reponse-en-anticorps/ 
debut infectant: 1, le plus jusqu'à 5, jusqu'à 28 mais peu infectant après 8.
 *)
(* les références ocaml sont appelées à varier *)

(* https://wwwnc.cdc.gov/eid/ serial interval 5.8 +- 1.5 *)
let debut_infectant = ref 1;;
let duree_infectant = ref 13;;
let duree_incubation = 5;; (* de 2 à 12, moyenne 5 https://www.pasteur.fr/fr/centre-medical/fiches-maladies/maladie-covid-19-nouveau-coronavirus *)
let duree_malade = 20;; (* un peu au pif *)
let mortalite = ref 0.02;;

(* R0: taux de reproduction de base *)
let voisins_contamines = ref 4.5;;
let vc_confinement = ref 2.;;(* mardi  10 mars *)
let div_vc_avant_confinement = ref 4.;; (* facteur de réduction de R0 8 jours avant le confinement *)
let decal_barriere = ref 8;; (* le lundi 9 mars *)

(**********************************************************************)
(* le pays est une matrice de coté ncase (env. 8000),
   une case par habitant, qui contient son état *)

let ncase = i_f (sqrt (f_i population)) + 1;;
let cote = f_i largeur_pays /. f_i ncase;;
let mat = Array.make_matrix ncase ncase 0;;
let dvoisins = ref (i_f (sqrt (f_i voisins) /. 2.));; (* = 285 *)

(* choisit un voisin au hasard avec distance 1 < dvoisins, rend ses coordonnées *)
let voisin_hasard mat x y =
  (min (ncase-1) (max 0 (x - !dvoisins + Random.int (2 * !dvoisins+1))),
   min (ncase-1) (max 0 (y - !dvoisins + Random.int (2 * !dvoisins+1))))
;;

let peut_mourir e =
  (* incubation entre 2 et 12 , moyenne 5, proba 2/3 entre 2 et 5, proba 1/3 entre 5 et 12 *)
  let dinc = if Random.int 3 <= 1
             then 5 - Random.int 4
             else 5 + Random.int 8 in
  let dm2 = duree_malade/2 in
  let edinc = e - dinc - dm2 in
  edinc > 0 && edinc <= dm2;;

let mort = -1;;

let plus_infectant_ni_malade e =
  e = mort || (e > !debut_infectant + !duree_infectant
               && e > 12 + duree_malade) (* incubation <= 12 *)

(* la liste des gens qui ont été en contact avec le virus,
 croissante pour l'inclusion *)
let lcontacts = Array.make (60*million) (0,0);;
let ncontacts = ref 0;;
let ajoute_contact x =
  lcontacts.(!ncontacts) <- x;
  ncontacts := !ncontacts + 1;
;;
let debut_contacts = ref 0;;
let lmorts = Array.make (1*million) (0,0);;
let nmorts = ref 0;;
let ajoute_mort x =
  lmorts.(!nmorts) <- x;
  nmorts := !nmorts + 1;
;;
let t0 = ref 0.;;

(**********************************************************************)
(* calcule les états des gens du pays le lendemain
met à jour le tableau des gens contaminés *)

let deja_infecte = ref 0;; (* infection d'un déjà infecté *)

(* le nombre de voisins infectés par jour suit une loi de Poisson
d'espérance !voisins_contamines / !duree_infectant *)

(* variable aléatoire de loi de Poisson et d'espérance lambda *)
let poisson lambda =
  let s = ref lambda in
  let r = ref (-1) in
  while !s >= 0. do
    r := !r + 1;
    s := !s +. log (1. -. Random.float 1.); (* pour pas avoir log(0) *)
  done;
  !r
;;

(* on va pas recalculer ca à chaque fois, ya une boucle, on va mémoriser:
  memo var de Poisson d'espérances de 0 (exclus) à max_esp (inclus) avec npas d'échantillonnage *)
let maxpoisson = 10000;;
let npas = 1000;;
let max_esp = 10.;;
let npas_max_esp = f_i npas /. max_esp;;
let mpoisson = Array.make_matrix (npas + 1) maxpoisson 0;;

for l = 1 to npas do
  for k = 0 to maxpoisson-1 do
    mpoisson.(l).(k) <- poisson (f_i l /. npas_max_esp);
  done;
done
;;

let poissonm lambda =
  mpoisson.(i_f (lambda *. npas_max_esp)).(Random.int maxpoisson)
;;

let etat (x,y) =
  mat.(x).(y)
;;

let set_etat (x,y) e =
  mat.(x).(y) <- e
;;

let propage () =
  (* voisins contamines durant une journée
     loi de Poisson de paramètre (espérance) !voisins_contamines / !duree_infectant  *)
  let vcontamine1 = !voisins_contamines /. f_i !duree_infectant in
  (* proba de mourir chaque jour entre duree_malade/2 et duree_malade: 
     suite de duree_malade/2 variables indépendantes et de même loi de Bernoulli *)
  let poisson_esp = mpoisson.(i_f (vcontamine1 *. npas_max_esp)) in
  let pmourir = 1. -. (1. -. !mortalite)**(1. /. (f_i duree_malade /. 2.)) in
  let lmodifs = ref [] in
  let fin_infectant = !debut_infectant + !duree_infectant in
  (* pour chaque personne en contactée par le virus *)
  for i = !debut_contacts to !ncontacts - 1 do
    let s = lcontacts.(i) in 
    let e = etat s in
    if e <> mort
    then
      if e <> 0 
      then
        if peut_mourir e && Random.float 1. < pmourir
        then lmodifs := (s,mort) :: !lmodifs
        else 
          (if e > !debut_infectant && e <= fin_infectant
           then 
             (let nvc = poisson_esp.(Random.int maxpoisson) in
              for k = 1 to nvc do
                (let v =
                   if Random.float 1. > !proba_voyageur (* pas voyageur*)
                   then let xc,yc = s in
                        voisin_hasard mat xc yc
                   else (Random.int ncase, Random.int ncase) (* voyageur *)
                 in if etat v = 0
                    then (lmodifs := (v,1) :: !lmodifs;
                          ajoute_contact v)
                    else (deja_infecte := !deja_infecte + 1))
              done;
             );
           lmodifs := (s, etat s + 1) :: !lmodifs);
  done;
  List.iter (fun (s,e) -> if e = mort then ajoute_mort s;
                          set_etat s e)
    !lmodifs;
  while plus_infectant_ni_malade (etat lcontacts.(!debut_contacts))
        && !debut_contacts <= !ncontacts do
    debut_contacts := !debut_contacts + 1;
  done;
;;

(**********************************************************************)
(* calcul d'erreur d'une simulation, en % *)

let erreur_hist hist jconf =
  let jdebut = jconf - !debut_confinement in (* = 0 normalement *)
  try
    let s = ref 0. in
    let compte = ref 0 in
    for j = jdebut to jdebut + Array.length !pays_morts - 1   do
      let morts,_,_,_ = hist.(j) in
      let jp = !debut_confinement + j - jconf in
      if !pays_morts.(jp) > 100
      then (compte := !compte + 1;
            s := !s +. (f_i morts -. f_i !pays_morts.(jp))**2.);
    done;
    let e = sqrt (!s /. f_i !compte) in
    (100. *. e /. f_i !pays_morts.(Array.length !pays_morts - 1) ,jdebut)
  with _ -> (0.,jdebut)
;;

let erreur_max = ref 20.;;
let erreur_max_pour_limite = ref 2.;;

(**********************************************************************)
(* calcule une simulation sur kmax jours au plus *)

type arret = {jour:int;jconf:int;morts:int;mortsfrance:int;erreur:float;raison:string};;

exception Arret of arret;;

let init_simul kmax =
  let hist = Array.make (kmax+1) (0,0,0,0) in
  ncontacts := 0;
  debut_contacts := 0;
  nmorts := 0;
  (* initialisation de la matrice *)
  for x = 0 to ncase - 1 do
    for y = 0 to ncase - 1 do
      mat.(x).(y) <- 0;
    done;
  done;
  (* les contaminés du début *)
  let ns = 3 in
  for g = 0 to ns - 1 do
    let s = (Random.int ncase, Random.int ncase) in
    set_etat s (!debut_infectant + g);
    hist.(g) <- (0,ns,ns,0);
    ajoute_contact s;
  done;
  t0 := Sys.time ();
  hist
;;

let verifie_temps j jconf nmorts =
  if Sys.time () -. !t0 > 180. 
  then raise (Arret {jour = !j;jconf = !jconf;morts = !nmorts;
                     mortsfrance = -1;erreur = 0.;raison = "temps limite dépassé"})
;;

let verifie_morts j jconf jp pays_morts db =
  if !jconf <> -1 && 0 <= !jp && !jp < Array.length !pays_morts
     && !nmorts > 100 && f_i !nmorts /. f_i !pays_morts.(!jp) < 0.5
  then raise (Arret {jour = !j;jconf = !jconf;morts = !nmorts;
                     mortsfrance = !pays_morts.(!jp);erreur = 0.;raison = "pas assez de morts"});
  if !jconf <> -1 && 0 <= !jp && !jp < Array.length !pays_morts
     && !nmorts > 100 && f_i !nmorts /. f_i !pays_morts.(!jp) > 1.5
  then raise (Arret {jour = !j;jconf = !jconf;morts = !nmorts;
                     mortsfrance = !pays_morts.(!jp);erreur = 0.;raison = "trop de morts"});
  if !jconf = -1 && f_i !nmorts /. f_i !pays_morts.(db) > 2.
  then raise (Arret {jour = !j;jconf = !jconf;morts = !nmorts;
                     mortsfrance = !pays_morts.(db);erreur = 0.;
                     raison = "trop de morts avant mesures barrières"})
;;

let verifie_erreur_apres_confinement jp debut_confinement pays_morts nmorts j jconf =
  if !jp > !debut_confinement + 4 
     && !jp < Array.length !pays_morts
     && abs_float (f_i !nmorts /. f_i !pays_morts.(!jp) -. 1.)
        > 2. *. !erreur_max /. 100.
  then raise (Arret {jour = !j;jconf = !jconf;morts = !nmorts;
                     mortsfrance = !pays_morts.(!jp);
                     erreur = abs_float (f_i !nmorts /. f_i !pays_morts.(!jp) -. 1.);
                     raison = "erreur trop grande un jour après confinement + 4"})
;;

let verifie_erreur_futur e j jconf nmorts pays_morts jp =
    if e >= 2. *. !erreur_max
    then raise (Arret {jour = !j;jconf = !jconf;morts = !nmorts;
                       mortsfrance = !pays_morts.(!jp);
                       erreur = e;
                       raison = "erreur moyenne trop grande"})
;;

let info_simul_futur j nmorts ncontacts deja_infecte mortalite =
  pr "jour %d, morts: %d, contacts: %d, en cours: %d, deja_infecte: %d, \nmortalite prévue: %.4f,morts/contacts: %.4f\n"
    !j !nmorts !ncontacts (!ncontacts - !debut_contacts) !deja_infecte !mortalite (f_i !nmorts /. f_i !ncontacts);
  flush stdout
;;

let info_simul_apres_conf j nmorts pays pays_morts jp ncontacts deja_infecte mortalite =
  pr "jour %d, morts: %d, %s: %d, contacts: %d, deja_infecte: %d, \nmortalite prévue: %.4f, morts/contacts: %.4f\n"
    !j !nmorts !pays !pays_morts.(!jp) !ncontacts !deja_infecte
    !mortalite (f_i !nmorts /. f_i !ncontacts) ;
  flush stdout
;;

let test_propage jconf pays_morts hist jp =
  !jconf = -1
  || (!jconf <> -1 &&  (not !arret_au_present
                        || !jp < Array.length !pays_morts + 10
                        || (!jp >= Array.length !pays_morts
                            && (let err,_ = erreur_hist hist !jconf in
                                err) < !erreur_max)))
;;

let test_confinement jconf nmorts pays_morts db =
  !jconf = -1
  && i_f (f_i !nmorts) >= (!pays_morts.(db) + !pays_morts.(db - 1)) / 2
  && i_f (f_i !nmorts) <= (!pays_morts.(db) + !pays_morts.(db + 1)) / 2
;;

let jour kmax =
  let hist = init_simul kmax in
  let jconf = ref (-1) in (* le jour du confinement *)
  let j1 = ref 0 in (* si arrêt de la simulation *)
  let k = ref kmax in
  let j = ref 1 in
  let r0 = !voisins_contamines in
  let pvoy0 = !proba_voyageur in
  let errmin = ref 100000000000000000000000000. in
  while !j <= !k do
    let jp = ref (!debut_confinement + !j - !jconf) in
    if test_propage jconf pays_morts hist jp
    then (propage (); flush stdout;

          j1 := !j;
          if !j < 120 then verifie_temps j jconf nmorts;
          let db = !debut_confinement - !decal_barriere in
          hist.(!j) <- (!nmorts,!ncontacts,0,0);
          (* on détecte 8 jours avant le confinement en france *)
          if test_confinement jconf nmorts pays_morts db
          then (jconf := !j + !decal_barriere;
                jp := !debut_confinement + !j - !jconf;
                voisins_contamines := !voisins_contamines /. !div_vc_avant_confinement;
                pr "\n######################## mesures barrières: %d, confinement: %d\n" !j !jconf);
          verifie_morts j jconf jp pays_morts db;
          if !jconf = -1
          then (pr "%d(m:%d,i:%d), " !j !nmorts !ncontacts; flush stdout)
          else (jp := !debut_confinement + !j - !jconf;
                verifie_erreur_apres_confinement jp debut_confinement pays_morts nmorts j jconf;
                if !j = !jconf 
                then (pr "######################## confinement %d\n" !jconf;
                      voisins_contamines := !vc_confinement;
                      proba_voyageur := !proba_voyageur /. !div_proba_voyageur;
                      dvoisins := !dvoisins / !div_dvoisins);
                if !j = !jconf + !decal_deconfinement
                then (pr "######################## déconfinement %d\n" !jconf;
                      voisins_contamines := sqrt (!vc_confinement *. (r0 /. !div_vc_avant_confinement));
                      proba_voyageur := (!proba_voyageur +. pvoy0) /. 2.;
                      dvoisins := !dvoisins * !div_dvoisins;
                      pr "R0=%.2f pvoy=%.4f dvoisins=%d\n" !voisins_contamines !proba_voyageur !dvoisins);
                if !jp >= Array.length !pays_morts (* le futur *)
                then (let e,decal = erreur_hist hist !jconf in
                      pr "erreur %.2f, decalage %d\n" e decal;flush stdout;
                      verifie_erreur_futur e j jconf nmorts pays_morts jp;
                      if e = !errmin && !jp >= Array.length !pays_morts + 3
                         && !arret_au_present (* arrêt de la simulation *)
                      (* si l'erreur est < 2%, on continue la simulation *)                             
                      then (if e < !erreur_max_pour_limite then k := 500 else k := 0);
                      if e < !errmin then errmin := e;
                      info_simul_futur j nmorts ncontacts deja_infecte mortalite)
                else info_simul_apres_conf j nmorts pays pays_morts jp ncontacts deja_infecte mortalite));
    j := !j + 1;
  done;
  if !jconf = -1
  then (pr  "============= pas arrivé au déconfinement: jour %d pas assez de morts: %d\n"
          !k !nmorts; flush stdout);
  hist,!jconf
;;

let info_params () =
  pr "datedebut = \"%s\"\njour = %d\nR0 = %.5f\ndR0 = %.5f\nR0confinement = %.5f\nprobavoyageur = %.5f\ndprobavoyageur = %.5f\ndebutinfectant = %d\ndureeinfectant = %d\nmortalite = %.5f\ndvoisins = %d\ndiv_dvoisins = %d\nerreur_max = %.2f\n" !debut_pays
    (Array.length !pays_morts) !voisins_contamines !div_vc_avant_confinement !vc_confinement
    !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite
    !dvoisins !div_dvoisins !erreur_max;
  flush stdout
;;

let limite_morts hist =
  let m = ref 0 in
  Array.iter( fun (x,_,_,_) -> if x <> 0 then m := max x !m) hist;
  !m
;;

(* carte des contaminés, à transformer en pdf avec prevision.py *)
let cree_carte file =
  let nc = Array.length mat / red_carte + 1 in 
  let carte = Array.make_matrix nc nc 0 in
  for x = 0 to ncase - 1 do
    for y = 0 to ncase - 1 do
      if etat (x,y) > 0 then let x,y = x/red_carte,y/red_carte in
                             carte.(x).(y) <- carte.(x).(y) + 1;
    done;
  done;
  let f = open_out file in
  Printf.fprintf f  "[";
  for x = 0 to nc-1 do
    Printf.fprintf f  "[";
    for y = 0 to nc-1 do
      Printf.fprintf f  "%d, " carte.(x).(y);
    done;
    Printf.fprintf f  "],\n";
  done;
  Printf.fprintf f  "]";
  close_out f;
  pr "fichier carte créé: %s\n" file;
;;

(***********************************************************************)
(* lance une simulation
   enregistre les résultats et une synthèse provisoire *)

let fichier_resultats = Printf.sprintf "%s/_resultats_jour_%d.csv" dir jourfr;;

let f = open_out fichier_resultats in
  Printf.fprintf f "%s" "erreur;R0;divR0;R01;pvoy;dpvoy;debi;duri;ifr;nc;lm;dv;ddv\n";
  close_out f
;;

let ajoute_resultat err ninfectes nmorts file =
  let f = open_out_gen [Open_append; Open_creat] 0o666 file in (* ouvrir en ajout *)
  Printf.fprintf f "%.5f;%.5f;%.5f;%.5f;%.5f;%.5f;%d;%d;%.5f;%d;%d;%d;%d\n"
            err !voisins_contamines !div_vc_avant_confinement !vc_confinement 
            !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite
            ninfectes nmorts !dvoisins !div_dvoisins;
  close_out f
;;

let nsimulations = ref 0;;
let erreur_min = ref 10000000000000000000.;;
let file_erreur_min = ref "";;
let nom_carte file = String.sub file 0 (String.length file - 3) ^ "_carte";; (* on enleve le .py *)

let ajour_synthese file =
  let carte_min = nom_carte !file_erreur_min in
  let carte = nom_carte file in
  cree_carte carte;
  List.iter
    (fun c -> pr "---------------------------- %s\n" c; flush stdout;
              let _ = Sys.command c in ())
    ["python3 cree_pdf.py " ^ file;
     "python3 prevision.py " ^ dir ^ " 20 2"; (* calcule l'histogramme *)
     "python3 cree_carte.py " ^ carte];
  if !arret_au_present (* sinon c'est qu'on calcule la limite *)
  then (if !pays = "france_rea" 
        then let _ = Sys.command ("python3 synthese.py "
                                  ^ " " ^ (string_of_int jourfr)
                                  ^ " " ^ (string_of_int !nsimulations)
                                  ^ " " ^ dir ^ "/" ^ "_intervalles.csv"
                                  ^ " " ^ fichier_resultats
                                  ^ " " ^ !file_erreur_min ^ ".png"
                                  ^ " " ^ carte_min ^ ".png") in
             ())
  else (* on calcule la limite *)
    (let f = open_out (Printf.sprintf "%s/_limites.csv" dir)  in
     Printf.fprintf f "%s\n%s;%d\n%s;%d\n" !file_erreur_min "limite contacts" !ncontacts "limite morts" !nmorts;
     close_out f)
;;

let simulations () =
  nsimulations := !nsimulations + 1;
  deja_infecte := 0;
  let jourfr = Array.length !pays_morts in
  let vc,vc1,pvoy,dvois = (* vc, pvoy, dvois changent pendant la fonction jour *)
    !voisins_contamines,!vc_confinement,!proba_voyageur,!dvoisins in
  pr "paramètres avant simulation:\n"; info_params ();
  let hist,jconf = jour nombre_de_jours_a_simuler in
  voisins_contamines := vc; (* on remet la valeur initiale *)
  proba_voyageur := pvoy; (* on remet la valeur initiale *)
  dvoisins := dvois; (* on remet la valeur initiale *)
  pr "paramètres après simulation:\n"; info_params ();
  let err, decal = erreur_hist hist jconf in
  (*let err = erreur hist jconf in*)
  if err < 0.00001 || jconf = -1
  then (pr "================================ trop ou pas assez de morts\n";
        flush stdout;
        failwith "trop ou pas assez de morts")
  else (let dcal = jconf - !debut_confinement - decal in
        pr "+++++++++++++++++++++++++++ decalage: %d\n" dcal;
        let jconf = !debut_confinement + decal in
        pr "erreur moyenne par jour:%.2f\n" err; flush stdout;
        let lm = limite_morts hist in
        let file = Printf.sprintf "%s/jour_%d_err_%.4f_R0_%.2f_dR0_%.1f_R01_%.2f_pvoy_%.2f_dpvoy_%.0f_debi_%d_duri_%d_mor_%.5f_dc_%d_nc_%d_lm_%d_dv_%d_ddv_%d.py"
                     dir jourfr err vc !div_vc_avant_confinement vc1
                     !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant
                     !mortalite dcal
                     !ncontacts lm
                     !dvoisins !div_dvoisins in
        pr "-------------------------------------- fichier %s" (file^"\n");
        if err < 2. *. !erreur_max
        then (
          let f = open_out file in
          Printf.fprintf f "pays = \"%s\"\ndatedebut = \"%s\"\njour = %d\nR0 = %.5f\ndR0 = %.5f\nR0confinement = %.5f\nprobavoyageur = %.5f\ndivprobavoyageur = %.5f\ndebutinfectant = %d\ndureeinfectant = %d\nmortalite = %.5f\ncontamines = %d\nlimite_morts = %d\ndvoisins = %d\ndiv_dvoisins = %d\n"
            !pays !debut_pays 
            (Array.length !pays_morts) vc !div_vc_avant_confinement vc1
            !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite
            !ncontacts lm !dvoisins !div_dvoisins;
          Printf.fprintf f "%s" "hist = [\n";
          for j = 0 to Array.length hist - 1 do
            let morts,nencontacts,_,_ = hist.(j) in
            let jp = !debut_confinement + j - jconf in
            let mpays = if 0 <= jp && jp < Array.length !pays_morts
                        then !pays_morts.(jp)
                        else -1
            in Printf.fprintf f "(%d,%d,%d,%d,%d,%d),\n"
                 jp morts nencontacts 0 0 mpays;
          done;
          Printf.fprintf f "%s" "]\n";
          close_out f;
          let morts,nencontacts,_,_ = hist.(Array.length hist - 1) in
          ajoute_resultat err nencontacts morts fichier_resultats;);
        if err < !erreur_min then (erreur_min := err; file_erreur_min := file; ajour_synthese file);
        if err < 2. (* en % *) then ajour_synthese file;
        err)
;;

(* effectue k prévisions, rend la moyenne des erreurs *)
let simulations_n k =
  try (let s = ref 0. in
       let s2 = ref 0. in
       for i = 1 to k do
         let err = simulations () in
         if err > !erreur_max
         then (pr "@@@@@@@@@@@@@@@@@@@@@@@@@@@@mauvais paramètres, on arrête les frais\n"; flush stdout;
               failwith "stop");
         s := !s +. err;
         s2 := !s2 +. err*.err;
       done;
       let esp = !s /. (f_i k) in
       let esp2 = !s2 /. (f_i k) in
       let ecartype = sqrt (esp2 -. esp *. esp) in (* variance = E(X^2) - E(X)^2 *)
       (esp,ecartype))
  with Arret {jour = j;jconf = jconf;morts = nmorts;
              mortsfrance = mf;
              erreur = e;
              raison = r} -> (pr "jour %d, jconf %d, morts %d, mortsfrance %d, erreur %.4f, %s\n"
                                j jconf nmorts mf e r;
                              (-1.,0.))
     | _ -> (-1.,0.)
;;

(***********************************************************************)
(* exploration de paramètres au hasard *)

let temps_depart = Sys.time ();;

let set_intervals interv =
  Array.iter
    (fun (p,a,b) ->
      match p with
      | "R0" -> voisins_contamines := hasard_float_intervalle a b
      | "dR0" -> div_vc_avant_confinement := hasard_float_intervalle a b
      | "R01" -> vc_confinement := hasard_float_intervalle a b
      | "pvoy" -> proba_voyageur := hasard_float_intervalle a b
      | "dpvoy" -> div_proba_voyageur := hasard_float_intervalle a b
      | "debi" -> debut_infectant := hasard_int_intervalle (i_f a) (i_f b)
      | "duri" -> duree_infectant := hasard_int_intervalle (i_f a) (i_f b)
      | "ifr" -> mortalite :=  hasard_float_intervalle a b
      | "dv" -> dvoisins := hasard_int_intervalle (i_f a) (i_f b)
      | "ddv" -> div_dvoisins := hasard_int_intervalle (i_f a) (i_f b)
      | "err_max" -> erreur_max := a
      | _ -> ())
    interv;
  let f = open_out (dir ^ "/" ^"_intervalles.csv") in
  Printf.fprintf f "%s" ";min;max\n";
  Array.iter (fun (p,a,b) ->
      match p with
      | "R01" | "pvoy" -> Printf.fprintf f "%s;%.2f;%.2f\n" p a b
      | "R0" | "dR0" -> Printf.fprintf f "%s;%.1f;%.1f\n" p a b
      | "dpvoy" | "debi"| "duri" | "dv" | "ddv" | "err_max" -> Printf.fprintf f "%s;%.0f;%.0f\n" p a b
      | "ifr" | _ -> Printf.fprintf f "%s;%.4f;%.4f\n" p a b)
    interv;
  close_out f;
;;


for  k = 1 to 10000000000000000 do

  if !arret_au_present
  then (set_intervals
          (* avec ifr faible: entre 0.01% et 0.5%
         [|"R0",2.,15.;
           "dR0",1.5,5.; 
           "R01",0.5,1.3;
           "pvoy",0.01,0.20;
           "dpvoy",2.,100.;
           "debi",1.,15.;
           "duri",3.,30.;
           "ifr",0.0001,0.005;
           "dv",100.,500.;
           "ddv",2.,10.;
           "err_max",20.,20.|]*)
         (* avec les valeurs moyennes du jour 63 *)
         (* [|"R0",5.7,5.8;
            "dR0",3.7,3.8;
            "R01",1.04,1.05;
            "pvoy",0.07,0.08;
            "dpvoy",71.,72.;
            "debi",2.5,2.6;
            "duri",8.0,8.1;
            "ifr",0.016,0.017;
            "dv",176.,177.;
            "ddv",6.,7.;
            "err_max",20.,20.;|];*)
        [|"R0",2.,10.;
           "dR0",1.5,5.; 
           "R01",0.5,1.2;
           "pvoy",0.01,0.20;
           "dpvoy",2.,100.;
           "debi",1.,9.;
           "duri",3.,21.;
           "ifr",0.001,0.03;
           "dv",100.,500.;
           "ddv",2.,10.;
           "err_max",10.,10.|];
        erreur_max_pour_limite := 2.;
        decal_deconfinement := 56 (*11 mai *)
       (*[|"R0",2.,10.;
           "dR0",1.5,5.; 
           "R01",0.5,1.2;
           "pvoy",0.01,0.20;
           "dpvoy",2.,100.;
           "debi",1.,9.;
           "duri",3.,21.;
           "ifr",0.001,0.03;
           "dv",100.,500.;
           "ddv",2.,10.;
           "err_max",10.,10.|]*)
       )
  else (* calcul des limites de contamines et de morts *)
    (* changer le format de _estimations, aussi dans prevision.py, qui le génère *)
    (if !nsimulations > 10 then exit 1;
     let ls = read_file file_estimations in
     List.iter (fun s -> pr "%s\n" s) ls; 
     let l = List.map (fun s -> List.nth (split s ';') 0) ls in
     for i = 0 to List.length l / 2 do
       try
         let p = List.nth l (2*i) in
         let v = float_of_string (List.nth l (2*i+1)) in
         pr "%s %s %.4f\n" p (List.nth l (2*i+1)) v;
         match p with
         | "R0" -> voisins_contamines := v
         | "dR0" -> div_vc_avant_confinement := v
         | "R01" -> vc_confinement := v
         | "pvoy" -> proba_voyageur := v
         | "dpvoy" -> div_proba_voyageur := v
         | "debi" -> debut_infectant := i_f v
         | "duri" -> duree_infectant := i_f v
         | "ifr" -> mortalite :=  v
         | "dv" -> dvoisins := i_f v
         | "ddv" -> div_dvoisins := i_f v
         | _ -> ()
       with _ -> ()
     done;
     erreur_max := 30.);
  
  pr "--------------------------------------------------------------\n";
  pr "nombre de prévisions %d\n" !nsimulations;
  pr "temps par prévisions: %.4f\n" ((Sys.time () -. temps_depart) /. f_i !nsimulations);
  pr "coté: %.2f, ncase: %d, voisins: %d, dvoisins: %.1f km, pixel: %.1f km\n"
    cote ncase voisins (f_i !dvoisins *. cote)  (cote *. f_i red_carte); flush stdout;
  pr "--------------------------------------------------------------\n";
  info_params ();
  let err,ect = simulations_n 1 (* 20 *) in
  if err >= 0.
  then ((* let _ = Sys.command "python3 prevision.py" in *)
    pr "**************************************************************\n";
    pr "                    erreur moyenne: %.2f\n" err;
    pr "**************************************************************\n";
    flush stdout);
done;
;;

(***********************************************************************)
(* compilation

ocamlopt -o propage propage.ml
time ./propage

*)

