(**********************************************************************)
(* Simulation de l'épidémie de covid19 en France *)
(**********************************************************************)

let pays = ref "france_rea";;
let methode = ref "hasard";; (* "hasard" ou "systematique" *)

(**********************************************************************)
(* utilitaires *)

let f_i = float_of_int;;
let i_f = int_of_float;;


(***********************************************************************)
(* patients en réanimation *)

(* https://geodes.santepubliquefrance.fr/#c=indicator&f=0&i=covid_hospit.rea&s=2020-04-25&t=a01&view=map2 
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
    5334;5127;4967;4785;4641
  |]
;;

let france_rea_06 =
  [|(* 18 mars *)
    1;5;7;7;8;13;12;17;18;22;21;26;30;34;
    (* 1 avril *)
    41;61;65;77;78;80;81;87;84;83;
    (* 11 avril *)
    79;80;79;79;69;64;58;56;55;50;
    (* 21 avril *)
    48;48;48;51;47
  |];;

(* rea cumulés *)
let s = ref 0;;
for i = 0 to Array.length france_rea - 1 do
  s := !s + france_rea.(i);
  france_rea.(i) <- !s;
done;;

(* on extrapole le nombre de morts: environ 8% des rea, 
 on oublie les ehpad (conditions de transimission spéciales)
 pour avoir une valeur raisonnable de la mortalité*)
let france_rea = Array.map
                   (fun x -> i_f (f_i x *. 0.073 (* *. 19323. /. 11842. *) ))
                   france_rea;;
(* décalage de 1 jour dans les bilans journaliers : on a les chiffres de la veille *)
let confinement_france_rea = 1 + 13;; (* mardi  17 mars, 699 en réa *)
let debut_france_rea = "4 mars 2020";;

(***********************************************************************)
(* paramètres propres au pays *)

let pays_morts = ref france_rea;;
let debut_confinement = ref confinement_france_rea;;
let debut_pays = ref debut_france_rea;;

(***********************************************************************)
(* population *)

let million = 1000000;;
let n = 60*million;; (* france 66 millions, italie 60 millions *)

(***********************************************************************)
(* paramètres de la maladie
sources:
 https://fr.wikipedia.org/wiki/Maladie_à_coronavirus_2019 
 https://www.lemonde.fr/blog/realitesbiomedicales/2020/04/17/covid-19-interrogations-sur-lexcretion-du-virus-et-la-reponse-en-anticorps/ 
debut infectant: 1, le plus jusqu'à 5, jusqu'à 28 mais peu infectant après 8.
 *)
(* les variables qui sont des références sont appelées à varier *)

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
let decal_barriere = ref 8;;

(* nombre de voisins (rencontres) possibles pendant la periode contaminante,
disons 13 jours, fixe, en fait, 
pour ne pas recalculer ler maillage à chaque test de paramètres *)

let voisins = 1000*13;; (* 100 * duree_infectant *)
let m = 1000 ;; (* largeur en kilometres du carré représentant le pays *)
let proba_voyageur = ref 0.10;;
let div_proba_voyageur = ref 100.;; (* facteur de réduction après le confinement *)

(***********************************************************************)
(* utilitaires *)

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

(**********************************************************************)
(* place au hasard les gens dans le pays *)

let placement_geographique n voisins m =
  let cote = sqrt ((f_i (m * m * voisins)) /. (f_i n)) in
  let ncase = i_f ((f_i m) /. cote) + 1 in
  pr "coté: %.2f, ncase: %d, voisins: %d\n" cote ncase voisins; flush stdout;
  let pos = Array.make n (0,0) in
  for s = 0 to n-1 do
    pos.(s) <- (Random.int ncase, Random.int ncase);
  done;
  pr "%s\n" "positions créés";
  let matl = Array.make_matrix ncase ncase [] in
  for s = 0 to n-1 do
    let xc,yc = pos.(s) in
    matl.(xc).(yc) <- s::matl.(xc).(yc);
  done;
  let mat = Array.make_matrix ncase ncase [||] in
  for x = 0 to ncase-1 do
    for y = 0 to ncase-1 do
      mat.(x).(y) <- Array.of_list matl.(x).(y);
    done;
  done;
  pr "mat remplie\n"; flush stdout;
  (pos,mat,cote)
;;

(* choisit un voisin au hasard, dans la case ou les cases voisines *)
let voisin_hasard mat x y =
    let n = Array.length mat in
    let dd = [|(-1,-1);(-1,0);(-1,1);
               (0,-1);(0,0);(0,1);
               (1,-1);(1,0);(1,1)|] in
    let rec aux () =
      let dx,dy = dd.(Random.int 9) in
      let t = mat.(min (n-1) (max 0 (x+dx))).(min (n-1) (max 0 (y+dy))) in
      if t = [||]
      then aux ()
      else t.(Random.int (Array.length t))
    in aux ()
;;

let peut_mourir e =
  (* incubation entre 2 et 12 , moyenne 5 
     proba 2/3 entre 2 et 5, proba 1/3 entre 5 et 12 *)
  let dinc = if Random.int 3 <= 1
             then 5 - Random.int 4
             else 5 + Random.int 8 in
  let dm2 = duree_malade/2 in
  let edinc = e - dinc - dm2 in
  edinc > 0 && edinc <= dm2;;

let gueri e = e > duree_incubation + duree_malade;;

let mort = -1;;

let plus_infectant_ni_malade e =
  e = mort || (e > !debut_infectant + !duree_infectant
               && e > 12 + duree_malade) (* incubation <= 12 *)

(* placement des individus dans le carré, matrices de proximites *)

let pos,mat,cote = placement_geographique n voisins m;;

(* la liste des gens qui ont été en contact avec le virus,
 croissante pour l'inclusion *)
let lcontacts = Array.make (10*million) 0;;
pr "lcontacts créée\n";;
let ncontacts = ref 0;;
let ajoute_contact x =
  lcontacts.(!ncontacts) <- x;
  ncontacts := !ncontacts + 1;
;;
let debut_contacts = ref 0;;

let lmorts = Array.make (1*million) 0;;
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

(* exemple: espérance 2:
f_i (Array.fold_left (fun s x -> s + x) 0 mpoisson.(200)) /. 1000.;;
 *)

let propage etat =
  let n = Array.length pos in
  (* voisins contamines durant une journée
     loi de Poisson de paramètre (espérance) !voisins_contamines / !duree_infectant  *)
  let vcontamine1 = !voisins_contamines /. f_i !duree_infectant in
  (* proba de mourir chaque jour entre duree_malade/2 et duree_malade: 
     suite de duree_malade/2 variables indépendantes et de même loi de Bernouilli *)
  let poisson_esp = mpoisson.(i_f (vcontamine1 *. npas_max_esp)) in
  let pmourir = 1. -. (1. -. !mortalite)**(1. /. f_i (duree_malade / 2)) in
  let lmodifs = ref [] in
  let fin_infectant = !debut_infectant + !duree_infectant in
  (* pour chaque personne en contactée par le virus *)
  for i = !debut_contacts to !ncontacts - 1 do
    let s = lcontacts.(i) in 
    let e = etat.(s) in
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
                   then let xc,yc = pos.(s) in
                        voisin_hasard mat xc yc
                   else Random.int n (* voyageur *)
                 in if etat.(v) = 0
                    then (lmodifs := (v,1) :: !lmodifs;
                          ajoute_contact v)
                    else (deja_infecte := !deja_infecte + 1))
              done;
             );
           lmodifs := (s,etat.(s) + 1) :: !lmodifs);
  done;
  List.iter (fun (s,e) -> if e = mort
                          then ajoute_mort s;
                          etat.(s) <- e) !lmodifs;
  while plus_infectant_ni_malade etat.(lcontacts.(!debut_contacts))
        && !debut_contacts <= !ncontacts do
    debut_contacts := !debut_contacts + 1;
  done;
  etat
;;

(* la simulation s'arrête au présent *)
let arret_au_present = ref true;;

(**********************************************************************)
(* calcul d'erreur d'une simulation *)

let erreur_min_hist hist jconf =
  let jdebut = jconf - !debut_confinement in
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

let erreur_max = ref 10.;; (* 5% *)

(**********************************************************************)
(* calcule une simulation sur kmax jours au plus *)

let jour kmax =
  let echelle = f_i (60*million) /. f_i n in
  let hist = Array.make (kmax+1) (0,0,0,0) in
  let etat = ref (Array.make n 0) in
  ncontacts := 0;
  debut_contacts := 0;
  nmorts := 0;
  (* les contaminés du début *)
  let ns = 3 in
  for s = 0 to ns - 1 do
    !etat.(s) <- !debut_infectant + s;
    hist.(s) <- (0,ns,ns,0);
    ajoute_contact s;
  done;
  let jconf = ref (-1) in
  t0 := Sys.time ();
  let jstop = ref 0 in
  let j1 = ref 0 in
  try(
    let k = ref kmax in
    let j = ref 1 in
    let errmin = ref 100000000000000000000000000. in
    while !j <= !k do
      let jp = !debut_confinement + !j - !jconf in
      if !jconf = -1 || (!jconf <> -1
                         &&  (not !arret_au_present
                              || jp < Array.length !pays_morts + 10
                              || (jp >= Array.length !pays_morts
                                  && (let err,_ = erreur_min_hist hist !jconf in
                                      err) < !erreur_max)))
      then (
        etat := propage !etat;
        j1 := !j;
        if Sys.time () -. !t0 > 180.
        then (pr "========================================%s\n" "temps trop long";
              flush stdout;
              jstop := !j;
              failwith "trop long");
        let morts = !nmorts in
        let ninfectants = 0 in
        let nencontacts = !ncontacts in
        let gueris = 0 in
        hist.(!j) <- morts,nencontacts,ninfectants,gueris;
        (* on détecte 8 jours avant le confinement en france *)
        if !jconf = -1
           && i_f (f_i morts *. echelle) >= (!pays_morts.(!debut_confinement - !decal_barriere)
                                             + !pays_morts.(!debut_confinement  - !decal_barriere - 1)) / 2
           && i_f (f_i morts *. echelle) <= (!pays_morts.(!debut_confinement - !decal_barriere)
                                             + !pays_morts.(!debut_confinement  - !decal_barriere + 1)) / 2
        then (jconf := !j + !decal_barriere;
              voisins_contamines := !voisins_contamines /. !div_vc_avant_confinement;
              pr "######################## mesures barrières: %d, confinement: %d\n" !j !jconf);
        if !jconf <> -1
        then (let jp = !debut_confinement + !j - !jconf in
              if jp > !debut_confinement + (Array.length !pays_morts - !debut_confinement) / 4
                 && jp < Array.length !pays_morts
                 && abs_float (f_i morts /. f_i !pays_morts.(jp) -. 1.)
                    > 2. *. !erreur_max /. 100.
              then (pr "============== erreur trop grande sur le nombre de morts\n"; flush stdout;
                    jstop := !j;
                    failwith "trop de morts");
              if !j = !jconf 
              then (pr "######################## confinement %d\n" !jconf;
                    voisins_contamines := !vc_confinement;
                    proba_voyageur := !proba_voyageur /. !div_proba_voyageur);
              if jp >= Array.length !pays_morts
              then (let e,decal = erreur_min_hist hist !jconf in
                    pr "erreur %.2f, decalage %d\n" e decal;flush stdout;
                    if e >= 2. *. !erreur_max
                    then (pr "erreur trop grande\n"; failwith "erreur trop grande");
                    if e = !errmin && jp >= Array.length !pays_morts + 3
                       && !arret_au_present (* arrêt de la simulation *)
                    then k := 0;
                    if e < !errmin then errmin := e;
                   );
              try (pr "jour %d, morts: %d, %s: %d, contacts: %d, deja_infecte: %d, \nmortalite prévue: %.4f, morts/contacts: %.4f\n"
                     !j morts !pays !pays_morts.(jp) !ncontacts !deja_infecte
                     !mortalite (f_i morts /. f_i !ncontacts) ;
                   flush stdout)
              with _ -> pr "jour %d, morts: %d, contacts: %d, deja_infecte: %d, \nmortalite prévue: %.4f,morts/contacts: %.4f\n"
                          !j morts !ncontacts !deja_infecte !mortalite (f_i morts /. f_i !ncontacts) ;
                        flush stdout));
      j := !j + 1;
    done;
    if !jconf = -1
    then (pr  "============= jour %d pas assez de morts: %d\n" !k !nmorts;
          flush stdout);
    hist,!jconf,!ncontacts,!etat)
  with _ -> (pr  "j1 %d, jconf %d,au jour %d, morts: %d\n"
               !j1 !jconf !jstop !nmorts;
             flush stdout;
             hist,-1,!ncontacts,!etat)
;;

let info_params () =
  pr "datedebut = \"%s\"\njour = %d\nR0 = %.5f\ndR0 = %.5f\nR0confinement = %.5f\nprobavoyageur = %.5f\ndprobavoyageur = %.5f\ndebutinfectant = %d\ndureeinfectant = %d\nmortalite = %.5f\n" !debut_pays
    (Array.length !pays_morts) !voisins_contamines !div_vc_avant_confinement !vc_confinement
    !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite;
  flush stdout
;;

let jourfr = Array.length !pays_morts;;
let fichier_resultats =
  Printf.sprintf "resultats_%s_jour_%d_%s.csv" !pays jourfr !methode;;
pr "%s\n" fichier_resultats;;
try
  let f = open_out fichier_resultats in
  Printf.fprintf f "%s" "erreur;R0;div R0 avant;R0confinement;probavoyageur;div probavoy;debutinfectant;dureeinfectant;mortalite\n";
  close_out f
with _ -> ()
;;
let fichier_resultats2 =
  Printf.sprintf "resultats2_%s_jour_%d_%s.csv" !pays jourfr !methode;;
pr "%s\n" fichier_resultats2;;
try
  let f = open_out fichier_resultats2 in
  Printf.fprintf f "%s" "erreur;R0;div R0 avant;R0confinement;probavoyageur;div probavoy;debutinfectant;dureeinfectant;mortalite\n";
  close_out f
with _ -> ()
;;

let remplace_point s =
  for i = 0 to String.length s - 1 do
    if s.[i] = '.'
    then Bytes.set s i ',';
  done;
  s
;;

let ajoute_resultat err ect file =
  let f = open_out_gen [Open_append; Open_creat] 0o666 file in (* ouvrir en ajout *)
  let s = Printf.sprintf "%.5f;%.5f;%.5f;%.5f;%.5f;%.5f;%.5f;%d;%d;%.5f\n"
            err ect !voisins_contamines !div_vc_avant_confinement !vc_confinement 
            !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite in
  Printf.fprintf f "%s" (remplace_point s);
  close_out f
;;

let erreur_min = ref 10000000000000000000000.;;

let limite_morts hist =
  let m = ref 0 in
  Array.iter( fun (x,_,_,_) -> if x <> 0 then m := max x !m) hist;
  !m
;;

(* carte des contaminés, à visualiser avec prevision.py *)
let cree_carte file etat =
  let n = Array.length mat in
  let carte = Array.make_matrix n n 0 in
  for x = 0 to n-1 do
    for y = 0 to n-1 do
      Array.iter
        (fun s -> if etat.(s) > 0 then carte.(x).(y) <- carte.(x).(y) + 1)
        mat.(x).(y);
    done;
  done;
  let f = open_out file in
  Printf.fprintf f  "[";
  for x = 0 to n-1 do
    Printf.fprintf f  "[";
    for y = 0 to n-1 do
      Printf.fprintf f  "%d, " carte.(x).(y);
    done;
    Printf.fprintf f  "],\n";
  done;
  Printf.fprintf f  "]";
  close_out f;
;;

(***********************************************************************)
(* lance une simulation
enregistre les résultats dans un fichier du type

jour_51_err_1.4177_R0_9.88_dR0_4.4_R01_0.97_pvoy_0.10_dpvoy_21_debi_7_duri_3_mor_0.08048_dc_0_nc_298764_lm_16046.py

dans le répertoire du pays,

et dans les fichiers resultats, resultats2,
calcule les fichiers de courbes _tout.pdf, 20meilleures.pdf, l'image du pays *)

let nprevisions = ref 0;;

let previsions () =
  nprevisions := !nprevisions + 1;
  deja_infecte := 0;
  let jourfr = Array.length !pays_morts in
  let vc,vc1,pvoy = (* vc, pvoy, duri changent pendant la fonction jour *)
    !voisins_contamines,!vc_confinement,!proba_voyageur in
  if false then info_params ();
  let hist,jconf,ncontacts,etat = jour 500 (* 150 *) in
  voisins_contamines := vc; (* on remet la valeur initiale *)
  proba_voyageur := pvoy; (* on remet la valeur initiale *)
  let err, decal = erreur_min_hist hist jconf in
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
        let file = Printf.sprintf "%s/jour_%d_err_%.4f_R0_%.2f_dR0_%.1f_R01_%.2f_pvoy_%.2f_dpvoy_%.0f_debi_%d_duri_%d_mor_%.5f_dc_%d_nc_%d_lm_%d.py" !pays
                     jourfr err vc !div_vc_avant_confinement vc1
                     !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite dcal
                     ncontacts lm in
        (*pr "%s" (file^"\n");*)
        if err < 2. *. !erreur_max
        then (
          let f = open_out file in
          Printf.fprintf f "pays = \"%s\"\ndatedebut = \"%s\"\njour = %d\nR0 = %.5f\ndR0 = %.5f\nR0confinement = %.5f\nprobavoyageur = %.5f\ndivprobavoyageur = %.5f\ndebutinfectant = %d\ndureeinfectant = %d\nmortalite = %.5f\ncontamines = %d\nlimite_morts = %d\n"
            !pays !debut_pays 
            (Array.length !pays_morts) vc !div_vc_avant_confinement vc1
            !proba_voyageur !div_proba_voyageur !debut_infectant !duree_infectant !mortalite
            ncontacts lm;
          Printf.fprintf f "%s" "hist = [\n";
          for j = 0 to Array.length hist - 1 do
            let morts,nencontacts,ninfectants,gueris = hist.(j) in
            let jp = !debut_confinement + j - jconf in
            let mpays = if 0 <= jp && jp < Array.length !pays_morts
                        then !pays_morts.(jp)
                        else -1
            in Printf.fprintf f "(%d,%d,%d,%d,%d,%d),\n"
                 jp morts nencontacts ninfectants gueris mpays;
          done;
          Printf.fprintf f "%s" "]\n";
          close_out f);
        if err < 2. *. !erreur_max
        then ajoute_resultat err 0. fichier_resultats2;
        if err < !erreur_min
        then (erreur_min := err;
              let _ = Sys.command ("python3 cree_pdf.py " ^ file) in
              let _ = Sys.command ("python3 prevision.py " ^ !pays ^ " 20") in
              let carte = String.sub file 0 (String.length file - 3) ^ "_carte" in
              cree_carte carte etat;
              let _ = Sys.command ("python3 cree_carte.py " ^ carte) in ()
             );
        err)
;;

(* effectue k prévisions, rend la moyenne des erreurs *)
let previsions_n k =
  try (let s = ref 0. in
       let s2 = ref 0. in
       for i = 1 to k do
         let err = previsions () in
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
  with _ -> (-1.,0.)
;;

(**********************************************************************)
(* jeux de paramètres *)

let params_int = [|debut_infectant; duree_infectant|];;
let params_float = [|voisins_contamines; vc_confinement;
                   proba_voyageur; mortalite|];;

let set_params p =
  voisins_contamines := p.(0);
  vc_confinement := p.(1);
  proba_voyageur := p.(2);
  debut_infectant := i_f p.(3);
  duree_infectant := i_f p.(4);
  mortalite := p.(5);
;;
(***********************************************************************)
(* exploration de paramètres au hasard *)

for  k = 1 to 10000000000000000 do
  (* on teste au hasard en gardant les fourchette de pasteur02548181 pour R0 R01 et mor *)
(*
          voisins_contamines := hasard_float_intervalle 3.18 3.43;
          div_vc_avant_confinement := hasard_float_intervalle 1.5 5.;
          vc_confinement := hasard_float_intervalle 0.5 0.55;
          proba_voyageur := hasard_float_intervalle 0.01 0.20;
          div_proba_voyageur := hasard_float_intervalle 1. 100.;
          debut_infectant := hasard_int_intervalle 1 9;
          duree_infectant := hasard_int_intervalle 3 15;
          mortalite := hasard_float_intervalle 0.0028 0.0088;
          erreur_max := 20.;
          arret_au_present := true;
          *)
  (* ensuite on change R01 *)
(*
          voisins_contamines := hasard_float_intervalle 3.18 3.43;
          div_vc_avant_confinement := hasard_float_intervalle 1.5 5.;
          vc_confinement := hasard_float_intervalle 0.5 0.99;
          proba_voyageur := hasard_float_intervalle 0.01 0.20;
          div_proba_voyageur := hasard_float_intervalle 1. 100.;
          debut_infectant := hasard_int_intervalle 1 9;
          duree_infectant := hasard_int_intervalle 3 15;
          mortalite := hasard_float_intervalle 0.0028 0.0088;
          erreur_max := 20.;
          arret_au_present := true;
 *)
  (* exploration aléatoire 25 avril *)
(*
  voisins_contamines := hasard_float_intervalle 2. 10.;
  div_vc_avant_confinement := hasard_float_intervalle 1.5 5.;
  vc_confinement := hasard_float_intervalle 0.5 1.2;
  proba_voyageur := hasard_float_intervalle 0.01 0.20;
  div_proba_voyageur := hasard_float_intervalle 2. 100.;
  debut_infectant := hasard_int_intervalle 1 9;
  duree_infectant := hasard_int_intervalle 3 15;
  mortalite := hasard_float_intervalle 0.002 0.10;
  erreur_max := 20.;
  arret_au_present := true;
 *)
  (* on affine la zone R0>5 IFR<5 avec pour les autres les valeurs moyennes et ecartypes *)
(*
  voisins_contamines := hasard_float_intervalle 4.9 8.8;
  div_vc_avant_confinement := hasard_float_intervalle 2.4 4.2;
  vc_confinement := hasard_float_intervalle 0.7 1.06;
  proba_voyageur := hasard_float_intervalle 0.05 0.16;
  div_proba_voyageur := hasard_float_intervalle 22. 75.;
  debut_infectant := hasard_int_intervalle 2 5;
  duree_infectant := hasard_int_intervalle 4 12;
  mortalite := hasard_float_intervalle 0.002 0.05;
  erreur_max := 20.;
  arret_au_present := true;
 *)
  (* avec les paramètres moyens *)

  voisins_contamines := 7.1;
  div_vc_avant_confinement := 3.2;
  vc_confinement := 0.87;
  proba_voyageur := 0.10;
  div_proba_voyageur := 47.;
  debut_infectant := 4;
  duree_infectant := 9;
  mortalite := 0.028;
  erreur_max := 20.;
  arret_au_present := false;

  
  pr "--------------------------------------------------------------\n";
  pr "nombre de prévisions %d\n" !nprevisions;
  pr "--------------------------------------------------------------\n";
  info_params ();
  let err,ect = previsions_n 20 in
  if err >= 0.
  then ((* let _ = Sys.command "python3 prevision.py" in *)
    pr "**************************************************************\n";
    pr "                    erreur moyenne: %.2f\n" err;
    pr "**************************************************************\n";
    flush stdout;
    ajoute_resultat err ect fichier_resultats);
done;
;;

(***********************************************************************)
(* compilation

ocamlopt -o propage propage.ml
time ./propage

*)
