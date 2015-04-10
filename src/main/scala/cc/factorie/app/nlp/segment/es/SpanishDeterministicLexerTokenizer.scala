package cc.factorie.app.nlp.segment.es

import java.io.StringReader

import cc.factorie.app.nlp.segment.{DeterministicNormalizingTokenizer, IsSgmlTag, DeterministicLexerTokenizer, PlainNormalizedTokenString}
import cc.factorie.app.nlp.{Document, Token}

import scala.collection.mutable


/**
 * Created by pat on 4/10/15.
 */
object SpanishDeterministicLexerTokenizer extends DeterministicLexerTokenizer {

  val compoundBuffer = new mutable.Queue[(String, Int, Int, Boolean, Boolean, Boolean, Boolean)]()

  override def process(document: Document): Document = {
    for (section <- document.sections) {
      /* Add this newline to avoid JFlex issue where we can't match EOF with lookahead */
      val reader = new StringReader(section.string + "\n")
      val lexer =
      // here we make sure that if normalize = false, we really don't normalize anything
        if(normalize)
          new SpanishLexer(reader, tokenizeSgml, tokenizeNewline, tokenizeWhitespace, tokenizeAllDashedWords, abbrevPrecedesLowercase,
            normalizeQuote, normalizeApostrophe, normalizeCurrency, normalizeAmpersand, normalizeFractions, normalizeEllipsis,
            undoPennParens, unescapeSlash, unescapeAsterisk, normalizeMDash, normalizeDash, normalizeHtmlSymbol, normalizeHtmlAccent)
        else
          new SpanishLexer(reader, tokenizeSgml, tokenizeNewline, tokenizeWhitespace, tokenizeAllDashedWords, abbrevPrecedesLowercase,
            false, false, false, false, false, false, false, false, false, false, false, false, false)

      // tokenString, posStart, tokenLength, isSgml, isContraction, isVerb, isCompound
      var currentToken : (String, Int, Int, Boolean, Boolean, Boolean, Boolean) = lexer.yylex().asInstanceOf[(String, Int, Int, Boolean, Boolean, Boolean, Boolean)]
      while (currentToken != null){
        currentToken = if (currentToken._5) {processContraction(currentToken)}
//        else if (currentToken._6) {processVerb(currentToken)}
        else if (currentToken._7) {processCompound(currentToken)}
        else currentToken
        val tok = new Token(section, currentToken._2, currentToken._2 + currentToken._3)
        if(normalize && tok.string != currentToken._1) tok.attr += new PlainNormalizedTokenString(tok, currentToken._1)
        if(tokenizeSgml && currentToken._4) tok.attr += IsSgmlTag
        currentToken = if (compoundBuffer.isEmpty) lexer.yylex().asInstanceOf[(String, Int, Int, Boolean, Boolean, Boolean, Boolean)] else compoundBuffer.dequeue()
      }
      /* If tokenizing newlines, remove the trailing newline we added */
      if(tokenizeNewline) section.remove(section.tokens.length - 1)
    }
    if (!document.annotators.contains(classOf[Token]))
      document.annotators(classOf[Token]) = this.getClass
    document
  }



  /**
   * Handles contractions like del and al, marked by the lexer
   *
   * del =&gt; de + l =&gt; de + el
   * al =&gt; a + l =&gt; a + el
   * con[mts]igo =&gt; con + [mts]i
   *
   */
  def processContraction(token : (String, Int, Int, Boolean, Boolean, Boolean, Boolean)) : (String, Int, Int, Boolean, Boolean, Boolean, Boolean) = {
    val word = token._1
    val start = token._2

    val lowered = word.toLowerCase
    val (first, firstLength, second, secondLength) = if (lowered.equals("del") || lowered.equals("al")) {
      val lastChar = word.charAt(lowered.length() - 1)
      (word.substring(0, lowered.length() - 1), lowered.length() - 1, if (Character.isLowerCase(lastChar)) "el" else "EL", 2)
    }
    else if (lowered.equals("conmigo") || lowered.equals("consigo")) {
      (word.substring(0, 3), 3, word.charAt(3) + "í", 2)
    }
    else if (lowered.equals("contigo")) {
      (word.substring(0, 3), 3, word.substring(3, 5), 2)
    }
    else {
      throw new IllegalArgumentException("Invalid contraction provided to processContraction")
    }
    compoundBuffer.enqueue((second, start+firstLength, secondLength, false, false, false, false))
    (first, start, firstLength, false, false, false, false)
  }

//  /**
//   * Handles verbs with attached suffixes, marked by the lexer:
//   *
//   * Escribamosela =&gt; Escribamo + se + la =&gt; escribamos + se + la
//   * Sentaos =&gt; senta + os =&gt; sentad + os
//   * Damelo =&gt; da + me + lo
//   *
//   */
//  def processVerb(token : (String, Int, Int, Boolean, Boolean, Boolean, Boolean)) : (String, Int, Int, Boolean, Boolean, Boolean, Boolean) = {
//    Pair<String, List<String>> parts = verbStripper.separatePronouns(cl.word());
//    if (parts == null)
//      return cl;
//    for(String pronoun : parts.second())
//    compoundBuffer.add(copyCoreLabel(cl, pronoun));
//    token
//  }

  /**
   * Splits a compound marked by the lexer.
   */
  def processCompound(token : (String, Int, Int, Boolean, Boolean, Boolean, Boolean)) : (String, Int, Int, Boolean, Boolean, Boolean, Boolean) = {
    val word = token._1
    val start = token._2
    val parts = word.replaceAll("\\-", " - ").split("\\s+")
    //TODO theres probably a nice scala way to do this, oh well
    var st = start
    compoundBuffer ++= parts.map(part => {
      val len = part.length
      val next = (part, st, len, false, false, false, false)
      st += len
      next
    })
    compoundBuffer.dequeue()
  }
}

object SpanishTokenizerTest extends App{

  val text = "Benjamín se encuentra en la mitad de un sueño placentero cuando en él aparece algo desagradable, Benjamín se retuerce en la cama, tratando de salir de ese sueño. Al gritar entran a su habitación para ver que ocurre, le dicen que no pasa nada y le secan la frente sudorosa; comentándole que tuvo otra vez un mal sueño pero que ahora esta despierto y puede tranquilizarse.\n\nSi has sentido alguna vez algo parecido, es que has tenido una pesadilla, no eres el único; casi todos la tienen de vez en cuando, tanto adultos como niños.\n\nUna pesadilla es un mal sueño, puede hacer que sientas miedo, ansiedad o enojo, pero no son reales ni pueden hacerte daño.\n\nMucha gente sueña con la misma pesadilla reiteradas veces. Otras personas tienen pesadillas en la cual el contenido cambia pero el mensaje es el mismo.\n\nCuando duermes tu cerebro sigue funcionando. Pasa por diversas fases de sueño, incluido el sueño REM (movimientos oculares rápidos); se llaman así porque durante esta fase del sueño, tus ojos se mueven de un lado a otro mientras los párpados están cerrados. Durante el sueño REM, tienes sueños, y a veces, estos sueños pueden ser aterradores o tristes.\n\nCada 90 minutos, aproximadamente tu cerebro alterna entre el sueño no REM y el sueño REM. Si te despiertas en la noche durante la fase REM, será más fácil que recuerdes lo que estabas soñando, esto es porque tus sueños y pesadillas más vividos ocurren en las primeras horas de la mañana.\n\nTe has preguntado ¿Por qué tienes pesadillas?. Bueno las situaciones estresantes que se producen durante el día pueden convertir los sueños en pesadillas. Las pesadillas pueden ser una forma de liberar las tensiones diarias. Eso implica enfrentarse a las cosas. Algunas veces los cambios importantes, como mudarse de casa o la enfermedad o muerte de un ser querido, pueden causar estrés y dar lugar a pesadillas. A veces un factor externo , por ejemplo, el ruido que hace una moto en la calle puede provocarte una pesadilla, en estos casos las personas no son conscientes pero es el cerebro quien lo detecta, produciendo así un cambio brusco en tu sueño.\n\nMuchas personas nos preguntamos ¿cómo es posible prevenir las pesadillas?. Teniendo claro que es normal tener pesadillas de vez en cuando, existen algunas técnicas para controlarlas, en las cuales está: seguir una rutina de sueño sana, dormir con una cobija de tu elección, dejar la puerta abierta, etc.\n\nDebemos tener muy claro que las pesadillas no son reales ni pueden hacernos daño. Soñar con algo aterrador no significa que sucederá en la vida real. Tampoco significa que seas una mala persona que quiere hacer cosas malas.\n\nNo eres infantil por sentir miedo después de una pesadilla. A veces el simple hecho de hablar con alguien basta para olvidar lo acontecido.\n\nLas pesadillas pueden ser aterradoras un rato, pero ahora ya tienes más claro que son y que hacer....." +
    "¿por qué no vienes conmigo?" +
    "¿por qué no vienes anti-christ?" +
    "¿por qué no vienes too-far?" +
    "¿por qué no vienes contigo?"

  val doc = new Document(text)
  SpanishDeterministicLexerTokenizer.process(doc)

  val doc2 = new Document(text)
  DeterministicNormalizingTokenizer.process(doc2)

  for (i <- 0 until Math.min(doc.tokenCount, doc2.tokenCount)){
    val t1 = doc.tokens(i)
    val t2 = doc2.tokens(i)
    println(t1.string, t1.stringStart, t1.stringEnd, t2.string, t2.stringStart, t2.stringEnd)
  }
}
