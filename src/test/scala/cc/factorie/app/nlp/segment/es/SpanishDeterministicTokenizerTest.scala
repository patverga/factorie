package cc.factorie.app.nlp.segment.es

import cc.factorie.app.nlp.segment.DeterministicTokenizer
import cc.factorie.util.FastLogging
import org.junit.Test
import org.scalatest.junit.JUnitSuite

/**
 * Created by pv on 2/20/15.
 */
class SpanishDeterministicTokenizerTest extends JUnitSuite with FastLogging {
  @Test def anusBlaster() = {
    val poopString = "taco burrito spanish deberíamos viviría"

    val vagitation = "Los bosques ocupan 2/5 del territorio, distribuidos en tres pisos altitudinales: hasta los 1200 metros, destacan la  y el; hasta los 1600-1700 metros, predomina el;" +
      " y hasta los 2200-2300 metros, abunda el, sustituido en las cumbres por los prados alpinos."
    
    val vag2 = "Andorra no es un miembro de pleno derecho en la pero, desde Acord d'unió duanera goza de una relación " +
      "especial y es tratado como si fuera parte de ella para el comercio en bienes manufacturados (exonerados de impuestos) y como no miembro de la UE para los productos agrícolas."

    val vag3 = "Descubrimiento de las islas.\nLas bulas \"Inter Caetera\" y \"Dudum si Quidem\" de 1493 le adjudicaban al Reino de España «todas aquellas islas y tierras firmes, encontradas y que se encuentren, descubiertas y que se descubran hacia el mediodía», fijada en una línea a cien leguas de las islas Azores. Las islas Malvinas estaban incluídas en las zonas aludidas por las bulas.\nLa Monarquía Hispánica proveyó una escuadra para ir a las Islas de las especias, siempre que se hallaran comprendidas dentro de las demarcaciones españolas y sin \"tocar\" en las portuguesas. Fue llamada en su día la “Armada de la Especiería” o “Armada de Magallanes”. Las singladuras se concluyeron con la primera vuelta al mundo (1519-1522) siendo la Nao \"Victoria\" la única nave que completó dicho periplo.\nLa expedición estaba formada por cinco naves con 234 hombres y zarpó de Sanlúcar de Barrameda, en la provincia de Cádiz, el 20 de septiembre de 1519.\nTras sublevarse contra Fernando de Magallanes el 1 de noviembre de 1520 (antes que la expedición llegue al estrecho de Magallanes y luego de estar en la costa de la actual provincia argentina de Santa Cruz), Gómez regresó a España por la ruta de Guinea y llegó a Sevilla el 6 de mayo de 1521. Fue sometido a un juicio en donde no se halla ninguna mención que pueda atribuirse a las Malvinas. Esteban Gómez cedió un indígena al cartógrafo Diogo Ribeiro en 1529, por lo que éste pudo haber obtenido de aquél un relato directo sobre las islas Sanson.\nOtra versión afín atribuye el descubrimiento al barco \"Victoria\", que fue enviado por Magallanes a rastrear al \"San Antón\" en aguas del océano Atlántico. Estos dos barcos fueron los únicos de la expedición de circunvalación que pudieron regresar a España.\nEn la cartografía de Diego de Ribero, la \"Carta universal en que se contiene todo lo que del mundo se ha descubierto fasta agora\" de 1527 y de 1529, pueden verse dos grupos de islas en la zona: las ocho o nueve islas llamadas \"Sanson\", a 49° S pero a mitad de camino de la costa que las Malvinas y las islas \"de los Patos\" muy cerca de la costa. En la actualización de 1533 ya no incluye a estas últimas. Mucho después continúan apareciendo las islas Sanson en los mapas de: Islario de Alonso de Santa Cruz (1541), Juan Bautista Agnese (1536-1545), Sebastián Caboto (1547), Darinel (1555), Diego Gutiérrez (1562), Bartolomé Olives (1562), Jorge Sideri (1563), Martínez (1577), José Rosacio (1580), etc., situadas siempre más al norte y más cerca de la costa que las Malvinas. En estas cartas las islas aparecen con las grafías de \"San Antón\", \"S. Antón\", \"Sansón\", \"Sanson\" o \"San Son\".\nNo se han encontrado menciones al avistaje en los relatos que se conservan del viaje: los diarios de Antonio Pigafetta, del piloto Francisco Albo, el Roteiro de un piloto genovés, ni la relación de Maximiliano de Transilvano. Aunque buena parte de los escritos de Magallanes se han perdido, y falta por completo la documentación de la nave de Juan Serrano y sus descripciones de San Julián al sur, la ausencia de referencias en las bitácoras citadas arrojó dudas sobre la veracidad de esta hipótesis. Sin embargo, en 1983 el historiador uruguayo Rolando Laguarda Trías encontró un documento en la Biblioteca Nacional de París, escrito por el fraile André Thevet en \"Le Gran Insulaire. Vol I\", fechado en 1586 (seis años antes del primer antecedente británico), que incluye un mapa en la página 229 donde aparecen “Les isles de Sansón ou des Geants” (las islas \"de Sansón\" o \"de los Gigantes\") en sorprendente concordancia geográfica con las islas Malvinas. Thevet menciona en el texto adjunto haber obtenido la posición y descripción del archipiélago de un piloto portugués miembro de la expedición de Magallanes, probablemente Álvaro de Mezquita, testigo directo del avistaje, con quien se entrevistó en Lisboa."
    println ("POOP STRING")
    println(SpanishDeterministicTokenizer(poopString).mkString(" "))
    println("SPANINSH VAG")
    println(SpanishDeterministicTokenizer(vag3).mkString(" "))
    println("NORMAL VAG")
    println(DeterministicTokenizer(vag3).mkString(" "))
  }

}
