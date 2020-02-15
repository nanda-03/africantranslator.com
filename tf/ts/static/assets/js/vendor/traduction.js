function getLinkByLanguages(input_language, output_language){
          return "https://tikquuss/"+input_language+"-"+output_language+"/model.json"
}

import * as tf from '@tensorflow/tfjs';
model_link = getLinkByLanguages(document.getElementById('left').innerHTML, document.getElementById('right').innerHTML)
const model = await tf.loadLayersModel(model_link);
example = document.getElementById('texte_depart').innerHTML;
const prediction = model.predict(example);

prefix_at = {'Francais':('/504/','.PDV2017'), "Anglais":('/1/','.NIV'),  "BIBALDA_TA_PELDETTA":('/557/','.MASSANA'), 
             'Bulu':("/856/",".BULU09"),  'Guiziga':('/550/','.BEGDC'), "Fulfulde_Adamaoua":("/906/",'.FB'),  "Fulfulde_DC":('/905/','.FBDC'), 
              'KALATA_KO_SC_Gbaya':('/871/','.GB11'), 'KALATA_KO_DC_Gbaya':('/868/','.GB11DC'), 'Kapsiki_DC':('/1515/','.KBDC'),
              'Tupurri':('/892/','.TUPB')}
prefix_nt = {'Bafia':('/876/',".BAFNT"), 'Dii':('/1511/','.DIINT'), 'Ejagham':('/889/','.ETUNT'), 'Ghomala':('/907/','.GNT02'),  
             'Vute':('/887/','.NTV'), 'Limbum':('/1240/','.LNT'), 'MKPAMAN_AMVOE_Ewondo':('/1854/','.NTE12'), 'Mofa':('/908/',".MAFNT"), 
             "Mofu_Gudur":('/551/',".MOF"), "Ngiemboon":('/229/','.NBGM'), 'Doyayo':('/880/','.NTD'), "Guidar":('/1120/',".GDRNT85"),
            'Peere_Nt&Psalms':('/1237/','.PERE'), 'Samba_Leko':('/1118/','.SLNT'), "Du_na_sdik_na_wiini_Alaw":('/1855/',".MUSGUM")}
function traduction(){
	if(prediction==""){
		document.getElementById('texte_arrivé').innerHTML=prediction;
	}else{
		document.getElementById('texte_arrivé').innerHTML=document.getElementById('texte_depart').innerHTML;
	}
	document.getElementById('texte_arrivé').innerHTML=document.getElementById('texte_depart').innerHTML;
}