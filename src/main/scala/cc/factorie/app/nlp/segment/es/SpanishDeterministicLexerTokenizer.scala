package cc.factorie.app.nlp.segment.es

import java.io.StringReader

import cc.factorie.app.nlp.segment.{IsSgmlTag, DeterministicLexerTokenizer, PlainNormalizedTokenString}
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
